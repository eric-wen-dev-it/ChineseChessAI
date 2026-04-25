using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System.Collections.Concurrent;
using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.MCTS
{
    public class BatchInference : IDisposable
    {
        private readonly CChessNet _model;
        private readonly int _batchSize;
        private readonly ConcurrentQueue<InferenceTask> _taskQueue = new();
        private readonly ManualResetEventSlim _signal = new(false);
        private readonly Task _workerTask;
        private readonly RuntimeDiagnostics.RollingCounter _queueDepthCounter;
        private readonly RuntimeDiagnostics.RollingCounter _batchSizeCounter;
        private readonly RuntimeDiagnostics.RollingCounter _batchLatencyMsCounter;
        private readonly RuntimeDiagnostics.RollingCounter _gpuWaitMsCounter;
        private long _lastMultiQueueTick;
        private volatile bool _isDisposed;

        private const int InputSize = 14 * 10 * 9;
        private static readonly TimeSpan OpportunisticBatchCoalescingWindow = TimeSpan.FromMilliseconds(2);
        private static readonly TimeSpan ConcurrentBatchCoalescingWindow = TimeSpan.FromMilliseconds(8);
        private static readonly TimeSpan ConcurrentSignalTtl = TimeSpan.FromMilliseconds(50);

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;
            string tag = $"{model.GetHashCode():x8}";
            _queueDepthCounter = new RuntimeDiagnostics.RollingCounter($"InferenceQueue/{tag}", 100);
            _batchSizeCounter = new RuntimeDiagnostics.RollingCounter($"InferenceBatch/{tag}", 50);
            _batchLatencyMsCounter = new RuntimeDiagnostics.RollingCounter($"InferenceLatencyMs/{tag}", 50);
            _gpuWaitMsCounter = new RuntimeDiagnostics.RollingCounter($"InferenceGpuWaitMs/{tag}", 50);
            _workerTask = Task.Run(() =>
            {
                try
                {
                    InferenceLoop();
                }
                catch
                {
                }
                finally
                {
                    // Worker exited (normally or via crash): drain any tasks that will never be processed.
                    while (_taskQueue.TryDequeue(out var task))
                        task.Tcs.TrySetCanceled();
                }
            });
        }

        public async Task<(float[] Policy, float Value)> PredictAsync(float[] inputData, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_isDisposed)
                return (new float[8100], 0f);

            var tcs = new TaskCompletionSource<(float[], float)>(TaskCreationOptions.RunContinuationsAsynchronously);
            _taskQueue.Enqueue(new InferenceTask(inputData, tcs, cancellationToken));
            int queueDepth = _taskQueue.Count;
            _queueDepthCounter.AddSample(queueDepth);
            if (queueDepth > 1)
            {
                Interlocked.Exchange(ref _lastMultiQueueTick, Stopwatch.GetTimestamp());
            }
            _signal.Set();

            using var reg = cancellationToken.UnsafeRegister(
                static (state, ct) => ((TaskCompletionSource<(float[], float)>)state!).TrySetCanceled(ct),
                tcs);
            return await tcs.Task;
        }

        private void InferenceLoop()
        {
            while (!_isDisposed)
            {
                _signal.Wait(100);

                if (_isDisposed)
                    break;

                _signal.Reset();
                if (_taskQueue.IsEmpty)
                    continue;

                var batchTasks = new List<InferenceTask>();
                if (_taskQueue.TryDequeue(out var firstTask))
                {
                    batchTasks.Add(firstTask);
                    CoalesceBatch(batchTasks);
                }

                CancelQueuedTasksThatNoLongerHaveCallers(batchTasks);
                if (batchTasks.Count > 0)
                {
                    ProcessBatch(batchTasks);
                }
            }

            while (_taskQueue.TryDequeue(out var task))
            {
                task.Tcs.TrySetResult((new float[8100], 0f));
            }
        }

        private void CoalesceBatch(List<InferenceTask> batchTasks)
        {
            if (batchTasks.Count >= _batchSize)
            {
                return;
            }

            TimeSpan waitWindow = HasRecentConcurrentSignal()
                ? ConcurrentBatchCoalescingWindow
                : OpportunisticBatchCoalescingWindow;

            var coalescingStopwatch = Stopwatch.StartNew();
            while (!_isDisposed && batchTasks.Count < _batchSize)
            {
                while (batchTasks.Count < _batchSize && _taskQueue.TryDequeue(out var task))
                {
                    batchTasks.Add(task);
                }

                if (batchTasks.Count >= _batchSize || coalescingStopwatch.Elapsed >= waitWindow)
                {
                    break;
                }

                Thread.Sleep(1);
            }
        }

        private static void CancelQueuedTasksThatNoLongerHaveCallers(List<InferenceTask> tasks)
        {
            for (int i = tasks.Count - 1; i >= 0; i--)
            {
                var task = tasks[i];
                if (!task.CancellationToken.IsCancellationRequested && !task.Tcs.Task.IsCompleted)
                {
                    continue;
                }

                if (task.CancellationToken.IsCancellationRequested)
                {
                    task.Tcs.TrySetCanceled(task.CancellationToken);
                }

                tasks.RemoveAt(i);
            }
        }

        private bool HasRecentConcurrentSignal()
        {
            long tick = Volatile.Read(ref _lastMultiQueueTick);
            if (tick == 0)
            {
                return false;
            }

            long elapsedTicks = Stopwatch.GetTimestamp() - tick;
            double elapsedSeconds = (double)elapsedTicks / Stopwatch.Frequency;
            return elapsedSeconds <= ConcurrentSignalTtl.TotalSeconds;
        }

        private void ProcessBatch(List<InferenceTask> tasks)
        {
            try
            {
                var totalStopwatch = Stopwatch.StartNew();
                if (_isDisposed)
                {
                    foreach (var task in tasks)
                        task.Tcs.TrySetException(new ObjectDisposedException(nameof(BatchInference)));
                    return;
                }

                _batchSizeCounter.AddSample(tasks.Count);
                var gpuWaitStopwatch = Stopwatch.StartNew();
                GpuExecutionGate.Run(() =>
                {
                    _gpuWaitMsCounter.AddSample(gpuWaitStopwatch.ElapsedMilliseconds);
                    _model.eval();
                    using var noGrad = torch.no_grad();

                    int batchCount = tasks.Count;
                    float[] batchData = new float[batchCount * InputSize];
                    for (int i = 0; i < batchCount; i++)
                    {
                        Buffer.BlockCopy(tasks[i].InputData, 0, batchData, i * InputSize * sizeof(float), InputSize * sizeof(float));
                    }

                    using var inputCpu = torch.tensor(batchData, new long[] { batchCount, 14, 10, 9 });
                    Tensor? inputGpu = null;
                    Tensor modelInput = inputCpu;

                    if (torch.cuda.is_available())
                    {
                        inputGpu = inputCpu.to(DeviceType.CUDA);
                        modelInput = inputGpu;
                    }

                    var (policyGpu, valueGpu) = _model.forward(modelInput);
                    try
                    {
                        using var policyCpu = policyGpu.cpu();
                        using var valueCpu = valueGpu.cpu();

                        for (int i = 0; i < batchCount; i++)
                        {
                            using var policyRow = policyCpu[i];
                            using var policyFloat = policyRow.to_type(ScalarType.Float32);
                            using var valueRow = valueCpu[i];
                            using var valueFloat = valueRow.to_type(ScalarType.Float32);

                            float[] policy = policyFloat.data<float>().ToArray();
                            float value = valueFloat.item<float>();
                            tasks[i].Tcs.TrySetResult((policy, value));
                        }
                    }
                    finally
                    {
                        valueGpu.Dispose();
                        policyGpu.Dispose();
                        inputGpu?.Dispose();
                    }
                });
                _batchLatencyMsCounter.AddSample(totalStopwatch.ElapsedMilliseconds);
            }
            catch (Exception ex)
            {
                foreach (var task in tasks)
                    task.Tcs.TrySetException(ex);
            }
        }

        public void Dispose()
        {
            _isDisposed = true;
            try
            {
                _signal.Set();
            }
            catch
            {
            }

            try
            {
                _workerTask.Wait(1000);
            }
            catch
            {
            }
        }

        private record InferenceTask(
            float[] InputData,
            TaskCompletionSource<(float[], float)> Tcs,
            CancellationToken CancellationToken);
    }
}
