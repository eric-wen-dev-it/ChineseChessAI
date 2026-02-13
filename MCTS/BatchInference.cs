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
        public async Task<(float[] Policy, float Value)> PredictAsync(torch.Tensor stateTensor)
        {
            var tcs = new TaskCompletionSource<(float[], float)>(); // 修改泛型
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
            // 核心修复：必须添加 DisposeScope，否则显存爆炸
            using (var scope = torch.NewDisposeScope())
            {
                using (var noGrad = torch.no_grad())
                {
                    var input = torch.cat(tasks.ConvertAll(t => t.Input), 0);
                    var (pLogits, vTensors) = _model.forward(input);

                    for (int i = 0; i < tasks.Count; i++)
                    {
                        // 核心修复：强制转成 C# 的 float 数组，完全切断 TorchSharp 句柄依赖
                        float[] policy = pLogits[i].to_type(ScalarType.Float32).data<float>().ToArray();
                        float value = vTensors[i].to_type(ScalarType.Float32).item<float>();

                        tasks[i].Tcs.SetResult((policy, value));
                    }
                }
            }
        }

        // 内部任务包装类
        private record InferenceTask(
                torch.Tensor Input,
                TaskCompletionSource<(float[], float)> Tcs); // 修改泛型
    }
}