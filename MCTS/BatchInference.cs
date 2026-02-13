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
            var tcs = new TaskCompletionSource<(float[], float)>();
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
                    ProcessBatch(batchTasks);
            }
        }

        private void ProcessBatch(List<InferenceTask> tasks)
        {
            _model.eval();

            // 核心修复：必须添加 DisposeScope，否则显存会无限增长直到崩溃
            using (var scope = torch.NewDisposeScope())
            {
                using (var noGrad = torch.no_grad())
                {
                    // 移除 AMP，使用标准 FP32 推理
                    var input = torch.cat(tasks.ConvertAll(t => t.Input), 0);
                    var (pLogits, vTensors) = _model.forward(input);

                    for (int i = 0; i < tasks.Count; i++)
                    {
                        // 立即转为 CPU 数组，切断 Tensor 句柄依赖
                        float[] policy = pLogits[i].to(ScalarType.Float32).data<float>().ToArray();
                        float value = vTensors[i].to(ScalarType.Float32).item<float>();

                        tasks[i].Tcs.SetResult((policy, value));
                    }
                }
            }
        }

        private record InferenceTask(torch.Tensor Input, TaskCompletionSource<(float[], float)> Tcs);
    }
}