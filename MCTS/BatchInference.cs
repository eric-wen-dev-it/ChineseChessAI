using ChineseChessAI.NeuralNetwork;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
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
        private volatile bool _isDisposed;

        private const int InputSize = 14 * 10 * 9;

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;
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
            _taskQueue.Enqueue(new InferenceTask(inputData, tcs));
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
                while (batchTasks.Count < _batchSize && _taskQueue.TryDequeue(out var task))
                {
                    batchTasks.Add(task);
                }

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

        private void ProcessBatch(List<InferenceTask> tasks)
        {
            try
            {
                if (_isDisposed)
                {
                    foreach (var task in tasks)
                        task.Tcs.TrySetException(new ObjectDisposedException(nameof(BatchInference)));
                    return;
                }

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

        private record InferenceTask(float[] InputData, TaskCompletionSource<(float[], float)> Tcs);
    }
}
