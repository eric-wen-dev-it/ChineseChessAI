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
        private readonly ConcurrentQueue<InferenceTask> _taskQueue = new();
        private readonly ManualResetEventSlim _signal = new(false); // 高灵敏度信号
        private volatile bool _isDisposed = false;

        private const int INPUT_SIZE = 14 * 10 * 9;

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;

            // 启动极速推理循环
            _ = Task.Run(() => {
                try { InferenceLoop(); }
                catch { }
            });
        }

        public async Task<(float[] Policy, float Value)> PredictAsync(float[] inputData)
        {
            if (_isDisposed)
                return (new float[8100], 0f);

            var tcs = new TaskCompletionSource<(float[], float)>(TaskCreationOptions.RunContinuationsAsynchronously);
            _taskQueue.Enqueue(new InferenceTask(inputData, tcs));
            _signal.Set(); // 唤醒推理线程
            
            return await tcs.Task;
        }

        private void InferenceLoop()
        {
            while (!_isDisposed)
            {
                // 等待信号，如果没有任务则进入休眠，不消耗 CPU
                _signal.Wait(100); 
                _signal.Reset();

                if (_taskQueue.IsEmpty) continue;

                var batchTasks = new List<InferenceTask>();
                
                // 极速收集任务，不进行任何异步切换
                while (batchTasks.Count < _batchSize && _taskQueue.TryDequeue(out var task))
                {
                    batchTasks.Add(task);
                }

                if (batchTasks.Count > 0)
                {
                    ProcessBatch(batchTasks);
                }
            }

            // 退出清理
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
                    foreach (var t in tasks) t.Tcs.TrySetException(new ObjectDisposedException(nameof(BatchInference)));
                    return;
                }

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
                foreach (var t in tasks) t.Tcs.TrySetException(ex);
            }
        }
        public void Dispose()
        {
            _isDisposed = true;
            try { _signal.Set(); } catch { } // 唤醒并退出，避免抛出异常
            // 不立刻调用 Dispose()，让 GC 自动回收，防止后台线程 Wait 被打断引发 ObjectDisposedException
        }

        private record InferenceTask(float[] InputData, TaskCompletionSource<(float[], float)> Tcs);
    }
}
