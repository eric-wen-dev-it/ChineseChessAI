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
            // 【优化】让 await 之后的代码在线程池运行，不要阻塞推理线程
            var tcs = new TaskCompletionSource<(float[], float)>(TaskCreationOptions.RunContinuationsAsynchronously);
            _taskQueue.Add(new InferenceTask(stateTensor, tcs));
            return await tcs.Task;
        }

        private void InferenceLoop()
        {
            while (true)
            {
                var batchTasks = new List<InferenceTask>();

                // 1. 尝试获取第一个任务（阻塞等待）
                if (_taskQueue.TryTake(out var firstTask, Timeout.Infinite))
                {
                    batchTasks.Add(firstTask);

                    // 2. 尝试获取后续任务以凑够 Batch（非阻塞）
                    while (batchTasks.Count < _batchSize && _taskQueue.TryTake(out var nextTask))
                    {
                        batchTasks.Add(nextTask);
                    }
                }

                if (batchTasks.Count > 0)
                {
                    // 【核心修复】添加 try-catch 块
                    try
                    {
                        ProcessBatch(batchTasks);
                    }
                    catch (Exception ex)
                    {
                        // 发生异常时，必须通知这一批的所有等待者，否则它们会永久卡死！
                        Console.WriteLine($"[BatchInference Critical Error] {ex.Message}");
                        foreach (var task in batchTasks)
                        {
                            // 将异常传递给 await 的调用方
                            task.Tcs.TrySetException(ex);
                        }

                        // 可选：如果遇到 OOM 等严重错误，可能需要清理显存或暂停一下
                        GC.Collect();
                    }
                }
            }
        }

        private void ProcessBatch(List<InferenceTask> tasks)
        {
            _model.eval();

            // 保持 DisposeScope，防止显存泄漏
            using (var scope = torch.NewDisposeScope())
            {
                using (var noGrad = torch.no_grad())
                {
                    // 合并 Batch
                    var input = torch.cat(tasks.ConvertAll(t => t.Input), 0);

                    // 前向推理
                    var (pLogits, vTensors) = _model.forward(input);

                    // 【核心修复】先将结果批量移动到 CPU，避免多次 GPU-CPU 通信开销，并防止非法访问
                    var pLogitsCPU = pLogits.cpu();
                    var vTensorsCPU = vTensors.cpu();

                    for (int i = 0; i < tasks.Count; i++)
                    {
                        // 从 CPU Tensor 读取数据是安全的
                        // 注意：这里 pLogitsCPU[i] 仍然是一个 Tensor 切片，需要确保数据类型正确
                        float[] policy = pLogitsCPU[i].to(ScalarType.Float32).data<float>().ToArray();

                        // item<float>() 对 CPU Tensor 也是安全的
                        float value = vTensorsCPU[i].to(ScalarType.Float32).item<float>();

                        tasks[i].Tcs.SetResult((policy, value));
                    }
                }
            }
        }

        private record InferenceTask(torch.Tensor Input, TaskCompletionSource<(float[], float)> Tcs);
    }
}