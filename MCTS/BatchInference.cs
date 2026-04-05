using ChineseChessAI.NeuralNetwork;
using System.Collections.Concurrent;
using TorchSharp;
using static TorchSharp.torch;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;

namespace ChineseChessAI.MCTS
{
    public class BatchInference : IDisposable
    {
        private readonly CChessNet _model;
        private readonly int _batchSize;
        private readonly BlockingCollection<InferenceTask> _taskQueue = new();
        private volatile bool _isDisposed = false;

        private const int INPUT_SIZE = 14 * 10 * 9;

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;

            // 启动纯逻辑循环，完全不使用任何 CancellationToken
            Task.Run(() => InferenceLoop());
        }

        public async Task<(float[] Policy, float Value)> PredictAsync(float[] inputData)
        {
            // 如果已经释放，直接返回空结果，绝不抛出异常
            if (_isDisposed)
                return (new float[8100], 0f);

            var tcs = new TaskCompletionSource<(float[], float)>(TaskCreationOptions.RunContinuationsAsynchronously);
            
            try
            {
                if (!_taskQueue.IsAddingCompleted)
                {
                    _taskQueue.Add(new InferenceTask(inputData, tcs));
                }
                else
                {
                    tcs.TrySetResult((new float[8100], 0f));
                }
            }
            catch
            {
                tcs.TrySetResult((new float[8100], 0f));
            }
            
            return await tcs.Task;
        }

        private void InferenceLoop()
        {
            try
            {
                // 纯布尔值控制，彻底杜绝 OperationCanceledException
                while (!_isDisposed)
                {
                    var batchTasks = new List<InferenceTask>();

                    try
                    {
                        // 10ms 轮询，保证即便在 CPU 满载时也能快速响应 Dispose 信号
                        if (_taskQueue.TryTake(out var firstTask, 10))
                        {
                            batchTasks.Add(firstTask);
                            while (batchTasks.Count < _batchSize && _taskQueue.TryTake(out var nextTask))
                            {
                                batchTasks.Add(nextTask);
                            }
                        }
                    }
                    catch { if (_isDisposed) break; continue; }

                    if (batchTasks.Count > 0)
                    {
                        ProcessBatch(batchTasks);
                    }
                }
            }
            finally
            {
                // 【终极加固】：退出循环时，给所有还在等待的 TCS 一个正常的结果 (Result)
                // 而不是 SetCanceled()，这样 await PredictAsync 的地方永远不会收到异常
                try
                {
                    while (_taskQueue.TryTake(out var task))
                    {
                        task.Tcs.TrySetResult((new float[8100], 0f));
                    }
                }
                catch { }
            }
        }

        private void ProcessBatch(List<InferenceTask> tasks)
        {
            try
            {
                if (_isDisposed) throw new ObjectDisposedException("engine");

                _model.eval();
                using var scope = torch.NewDisposeScope();
                using var noGrad = torch.no_grad();

                int batchCount = tasks.Count;
                float[] batchData = new float[batchCount * INPUT_SIZE];

                for (int i = 0; i < batchCount; i++)
                {
                    Buffer.BlockCopy(tasks[i].InputData, 0, batchData, i * INPUT_SIZE * sizeof(float), INPUT_SIZE * sizeof(float));
                }

                var inputCPU = torch.tensor(batchData, new long[] { batchCount, 14, 10, 9 });
                var inputGPU = torch.cuda.is_available() ? inputCPU.to(DeviceType.CUDA) : inputCPU;

                var (pLogitsGPU, vTensorsGPU) = _model.forward(inputGPU);

                var pLogitsCPU = pLogitsGPU.cpu();
                var vTensorsCPU = vTensorsGPU.cpu();

                for (int i = 0; i < batchCount; i++)
                {
                    float[] policy = pLogitsCPU[i].to(ScalarType.Float32).data<float>().ToArray();
                    float value = vTensorsCPU[i].to(ScalarType.Float32).item<float>();
                    tasks[i].Tcs.TrySetResult((policy, value));
                }
            }
            catch (Exception ex)
            {
                foreach (var t in tasks) t.Tcs.TrySetResult((new float[8100], 0f));
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;

            try
            {
                _taskQueue.CompleteAdding();
                // 彻底释放
                Task.Delay(10).ContinueWith(_ => _taskQueue.Dispose());
            }
            catch { }
        }

        private record InferenceTask(float[] InputData, TaskCompletionSource<(float[], float)> Tcs);
    }
}
