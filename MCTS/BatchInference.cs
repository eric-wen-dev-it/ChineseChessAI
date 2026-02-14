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

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;
            Task.Run(InferenceLoop);
        }

        public async Task<(float[] Policy, float Value)> PredictAsync(torch.Tensor stateTensor)
        {
            var tcs = new TaskCompletionSource<(float[], float)>(TaskCreationOptions.RunContinuationsAsynchronously);
            _taskQueue.Add(new InferenceTask(stateTensor, tcs));
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
            // _model 已经在 GPU 上了
            _model.eval();

            using (var scope = torch.NewDisposeScope())
            {
                using (var noGrad = torch.no_grad())
                {
                    // 【关键步骤】
                    // 1. 在 CPU 上合并所有小 Tensor (tasks[i].Input 都在 CPU)
                    var inputCPU = torch.cat(tasks.ConvertAll(t => t.Input), 0);

                    // 2. 统一移动到 GPU (仅发生一次 CPU->GPU 拷贝)
                    var inputGPU = torch.cuda.is_available() ? inputCPU.to(DeviceType.CUDA) : inputCPU;

                    // 3. 执行推理
                    var (pLogitsGPU, vTensorsGPU) = _model.forward(inputGPU);

                    // 4. 结果搬回 CPU
                    var pLogitsCPU = pLogitsGPU.cpu();
                    var vTensorsCPU = vTensorsGPU.cpu();

                    for (int i = 0; i < tasks.Count; i++)
                    {
                        // 5. 安全地转换为数组
                        float[] policy = pLogitsCPU[i].to(ScalarType.Float32).data<float>().ToArray();
                        float value = vTensorsCPU[i].to(ScalarType.Float32).item<float>();

                        tasks[i].Tcs.SetResult((policy, value));
                    }
                }
            }
        }

        private record InferenceTask(torch.Tensor Input, TaskCompletionSource<(float[], float)> Tcs);
    }
}