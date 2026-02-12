using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using ChineseChessAI.NeuralNetwork;

namespace ChineseChessAI.MCTS
{
    /// <summary>
    /// GPU 批处理推理器：将多个搜索线程的推理请求合并处理
    /// </summary>
    public class BatchInference
    {
        private readonly CChessNet _model;
        private readonly int _batchSize;

        // 线程安全队列，用于接收推理任务
        private readonly BlockingCollection<InferenceTask> _taskQueue = new();

        public BatchInference(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _batchSize = batchSize;

            // 启动后台 GPU 处理循环
            Task.Run(InferenceLoop);
        }

        /// <summary>
        /// 外部搜索线程调用此方法获取预测结果
        /// </summary>
        public async Task<(torch.Tensor Policy, float Value)> PredictAsync(torch.Tensor stateTensor)
        {
            var tcs = new TaskCompletionSource<(torch.Tensor, float)>();
            _taskQueue.Add(new InferenceTask(stateTensor, tcs));
            return await tcs.Task;
        }

        private void InferenceLoop()
        {
            while (true)
            {
                var batchTasks = new List<InferenceTask>();

                // 1. 等待并获取第一个任务
                if (_taskQueue.TryTake(out var firstTask, Timeout.Infinite))
                {
                    batchTasks.Add(firstTask);

                    // 2. 尽量填满 Batch
                    while (batchTasks.Count < _batchSize && _taskQueue.TryTake(out var nextTask))
                    {
                        batchTasks.Add(nextTask);
                    }
                }

                if (batchTasks.Count == 0)
                    continue;

                // 3. 执行 GPU 推理
                ProcessBatch(batchTasks);
            }
        }

        private void ProcessBatch(List<InferenceTask> tasks)
        {
            _model.eval();
            using (var noGrad = torch.no_grad())
            {
                // 将所有 Tensor 合并为一个大 Batch [Batch, 14, 10, 9]
                var input = torch.cat(tasks.ConvertAll(t => t.Input), 0);

                var (pLogits, vTensors) = _model.forward(input);

                // 4. 将结果分发回各自的 TaskCompletionSource
                for (int i = 0; i < tasks.Count; i++)
                {
                    // 克隆 Tensor 以防被后续释放，并转移到 CPU 方便主线程读取
                    var policy = pLogits[i].detach().cpu();
                    float value = vTensors[i].item<float>();

                    tasks[i].Tcs.SetResult((policy, value));
                }
            }
        }

        // 内部任务包装类
        private record InferenceTask(
            torch.Tensor Input,
            TaskCompletionSource<(torch.Tensor, float)> Tcs);
    }
}