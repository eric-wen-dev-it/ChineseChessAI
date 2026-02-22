using ChineseChessAI.NeuralNetwork;
using TorchSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace ChineseChessAI.Training
{
    public class Trainer
    {
        private readonly CChessNet _model;
        private torch.optim.Optimizer _optimizer;
        private LRScheduler _scheduler; // 学习率调度器实例
        private readonly double _learningRate = 0.001;

        // 记录迭代次数，用于触发学习率更新
        private int _iterationCount = 0;

        public Trainer(CChessNet model)
        {
            _model = model;
            ResetOptimizer();
        }

        public void ResetOptimizer()
        {
            var parameters = _model.parameters().Where(p => p.requires_grad).ToList();
            if (parameters.Count == 0)
                return;

            // 1. 初始化 Adam 优化器
            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);

            // 2. 初始化学习率调度器
            // 注意：由于引入了 Chunk 切片机制，每次调用 Train 只处理几千个样本。
            // 设定 step_size: 500 意味着每训练 500 个 Chunk (约两百万数据)，学习率变为原来的 50%
            _scheduler = StepLR(_optimizer, step_size: 500, gamma: 0.5);
        }

        /// <summary>
        /// 供 MainWindow 调用的多轮训练接口（完美对接 List<TrainingExample> 切片）
        /// </summary>
        public float Train(List<TrainingExample> examples, int epochs)
        {
            if (examples == null || examples.Count == 0)
                return 0f;

            // 1. 将 List 转换为 Tensor (安全！因为外部已经做了 Chunk 限制，这里不会爆显存)
            var statesList = new List<Tensor>(examples.Count);
            var policiesList = new List<Tensor>(examples.Count);
            var valuesList = new List<Tensor>(examples.Count);

            foreach (var ex in examples)
            {
                // 确保 state 的维度形状匹配模型输入（通常是 1x10x9）
                statesList.Add(tensor(ex.State).view(1, 10, 9));
                policiesList.Add(tensor(ex.Policy));
                valuesList.Add(tensor(ex.Value));
            }

            using var statesTensor = stack(statesList);
            using var policiesTensor = stack(policiesList);
            using var valuesTensor = stack(valuesList);

            double totalLoss = 0;

            // 2. 循环训练指定轮次
            for (int e = 0; e < epochs; e++)
            {
                totalLoss += TrainStep(statesTensor, policiesTensor, valuesTensor);
            }

            // 3. 释放临时列表中的 Tensor 防止内存泄漏
            foreach (var t in statesList)
                t.Dispose();
            foreach (var t in policiesList)
                t.Dispose();
            foreach (var t in valuesList)
                t.Dispose();

            // 4. 更新学习率调度器状态
            _iterationCount++;
            _scheduler.step();

            return (float)(totalLoss / epochs);
        }

        private double TrainStep(Tensor states, Tensor targetPolicies, Tensor targetValues)
        {
            using var scope = torch.NewDisposeScope();

            _model.train();
            _optimizer.zero_grad();

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            var x = states.to(device).to_type(ScalarType.Float32);
            var y_policy = targetPolicies.to(device).to_type(ScalarType.Float32);
            var y_value = targetValues.to(device).to_type(ScalarType.Float32);

            var (policyLogits, valuePred) = _model.forward(x);

            // === Loss 计算逻辑 ===
            // Value Loss: 均方误差 (MSE)
            var vLoss = torch.nn.functional.mse_loss(valuePred, y_value.view(-1, 1));

            // Policy Loss: 交叉熵 (CrossEntropy) 的手动实现
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(y_policy * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            if (!totalLoss.requires_grad)
            {
                throw new Exception($"计算图断裂！参数状态: {_model.parameters().First().requires_grad}");
            }

            // 反向传播与优化
            totalLoss.backward();
            _optimizer.step();

            return totalLoss.item<float>();
        }

        /// <summary>
        /// 获取当前实际运行中的学习率
        /// </summary>
        public double GetCurrentLR()
        {
            var groups = _optimizer.ParamGroups;
            if (groups != null && groups.Any())
            {
                return groups.First().LearningRate;
            }
            return _learningRate;
        }
    }
}