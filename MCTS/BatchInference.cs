using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using ChineseChessAI.NeuralNetwork;
using System.Linq; // 确保引用 Linq

namespace ChineseChessAI.MCTS
{
    public class BatchInference
    {
        private readonly CChessNet _model;
        private readonly int _batchSize;
        // 修改 Tcs 泛型为 float[]，彻底切断与 Tensor 的联系
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

            // 关键修复 1: 必须加 DisposeScope，否则显存几秒钟就会爆满
            using (var scope = torch.NewDisposeScope())
            {
                // 关键优化 2: 开启 AMP，让 4070 使用 Tensor Cores 加速推理
                using (var amp = torch.cuda.amp.autocast())
                using (var noGrad = torch.no_grad())
                {
                    var input = torch.cat(tasks.ConvertAll(t => t.Input), 0);
                    var (pLogits, vTensors) = _model.forward(input);

                    for (int i = 0; i < tasks.Count; i++)
                    {
                        // 关键修复 3: 立即转为 float[] 数组，数据回到 CPU，Tensor 在 scope 结束时销毁
                        float[] policy = pLogits[i].to(torch.ScalarType.Float32).data<float>().ToArray();
                        float value = vTensors[i].to(torch.ScalarType.Float32).item<float>();

                        tasks[i].Tcs.SetResult((policy, value));
                    }
                }
            }
        }

        private record InferenceTask(torch.Tensor Input, TaskCompletionSource<(float[], float)> Tcs);
    }
}