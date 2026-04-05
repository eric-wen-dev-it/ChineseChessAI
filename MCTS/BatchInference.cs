using ChineseChessAI.NeuralNetwork;
using System.Collections.Concurrent;
using TorchSharp;
using static TorchSharp.torch;
using System.Threading; // 添加此引用

namespace ChineseChessAI.MCTS
{
    public class BatchInference : IDisposable
    {
        private readonly CChessNet _model;
        private readonly int _batchSize;
        private readonly BlockingCollection<InferenceTask> _taskQueue = new();
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();

        private const int INPUT_SIZE = 14 * 10 * 9;

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;

            // 【核心修复】：传入取消令牌
            Task.Run(() => InferenceLoop(_cts.Token));
        }

        public async Task<(float[] Policy, float Value)> PredictAsync(float[] inputData)
        {
            if (inputData.Length != INPUT_SIZE)
                throw new ArgumentException($"输入数据长度错误，期望 {INPUT_SIZE}，实际 {inputData.Length}");

            var tcs = new TaskCompletionSource<(float[], float)>(TaskCreationOptions.RunContinuationsAsynchronously);
            _taskQueue.Add(new InferenceTask(inputData, tcs));
            return await tcs.Task;
        }

        private void InferenceLoop(CancellationToken token)
        {
            // 【核心修复】：循环条件监听取消令牌
            while (!token.IsCancellationRequested)
            {
                var batchTasks = new List<InferenceTask>();

                try
                {
                    // 使用超时时间防止死锁，以便定期检查 token
                    if (_taskQueue.TryTake(out var firstTask, 5, token))
                    {
                        batchTasks.Add(firstTask);
                        while (batchTasks.Count < _batchSize && _taskQueue.TryTake(out var nextTask))
                        {
                            batchTasks.Add(nextTask);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    break; // 收到取消信号，退出循环
                }

                if (batchTasks.Count > 0)
                {
                    try
                    {
                        ProcessBatch(batchTasks);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[BatchInference Critical Error] {ex.Message}");
                        foreach (var task in batchTasks)
                        {
                            task.Tcs.TrySetException(ex);
                        }
                        GC.Collect();
                    }
                }
            }
        }

        private void ProcessBatch(List<InferenceTask> tasks)
        {
            _model.eval();

            using (var scope = torch.NewDisposeScope())
            {
                using (var noGrad = torch.no_grad())
                {
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
                        tasks[i].Tcs.SetResult((policy, value));
                    }
                }
            }
        }

        // 实现 IDisposable 以便优雅关闭
        public void Dispose()
        {
            _cts.Cancel();
            _cts.Dispose();
            _taskQueue.Dispose();
        }

        private record InferenceTask(float[] InputData, TaskCompletionSource<(float[], float)> Tcs);
    }
}