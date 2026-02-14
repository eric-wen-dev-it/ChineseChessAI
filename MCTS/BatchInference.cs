using ChineseChessAI.NeuralNetwork;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.MCTS
{
    public class BatchInference
    {
        private readonly CChessNet _model;
        private readonly int _batchSize;
        private readonly BlockingCollection<InferenceTask> _taskQueue = new();

        // 预计算输入维度：14通道 * 10行 * 9列 = 1260
        private const int INPUT_SIZE = 14 * 10 * 9;

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;
            Task.Run(InferenceLoop);
        }

        // 修改：接收 float[] 数组，彻底切断 Tensor 的线程依赖
        public async Task<(float[] Policy, float Value)> PredictAsync(float[] inputData)
        {
            if (inputData.Length != INPUT_SIZE)
                throw new ArgumentException($"输入数据长度错误，期望 {INPUT_SIZE}，实际 {inputData.Length}");

            var tcs = new TaskCompletionSource<(float[], float)>(TaskCreationOptions.RunContinuationsAsynchronously);
            _taskQueue.Add(new InferenceTask(inputData, tcs));
            return await tcs.Task;
        }

        private void InferenceLoop()
        {
            while (true)
            {
                var batchTasks = new List<InferenceTask>();

                if (_taskQueue.TryTake(out var firstTask, Timeout.Infinite))
                {
                    batchTasks.Add(firstTask);
                    while (batchTasks.Count < _batchSize && _taskQueue.TryTake(out var nextTask))
                    {
                        batchTasks.Add(nextTask);
                    }
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
                    // 1. 构建 Batch 数据 (在 CPU 端拼装大数组)
                    int batchCount = tasks.Count;
                    float[] batchData = new float[batchCount * INPUT_SIZE];

                    // 使用 BlockCopy 极速拷贝
                    for (int i = 0; i < batchCount; i++)
                    {
                        Buffer.BlockCopy(tasks[i].InputData, 0, batchData, i * INPUT_SIZE * sizeof(float), INPUT_SIZE * sizeof(float));
                    }

                    // 2. 创建 Tensor 并一次性传输到 GPU
                    // 注意形状：[Batch, 14, 10, 9]
                    var inputCPU = torch.tensor(batchData, new long[] { batchCount, 14, 10, 9 });
                    var inputGPU = torch.cuda.is_available() ? inputCPU.to(DeviceType.CUDA) : inputCPU;

                    // 3. 推理
                    var (pLogitsGPU, vTensorsGPU) = _model.forward(inputGPU);

                    // 4. 结果搬回 CPU
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

        private record InferenceTask(float[] InputData, TaskCompletionSource<(float[], float)> Tcs);
    }
}