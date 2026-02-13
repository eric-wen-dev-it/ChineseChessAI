using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;
using ChineseChessAI.NeuralNetwork;

namespace ChineseChessAI.Training
{
    public class Trainer
    {
        private readonly CChessNet _model;
        private readonly torch.optim.Optimizer _optimizer;
        private readonly double _learningRate = 0.001;
        private readonly double _l2Reg = 1e-4; // L2 正则化系数

        public Trainer(CChessNet model)
        {
            _model = model;
            // 使用 Adam 优化器，这是 AlphaZero 变体中最常用的选择
            _optimizer = torch.optim.Adam(_model.parameters(), _learningRate, weight_decay: _l2Reg);
        }

        /// <summary>
        /// 执行一个训练步
        /// </summary>
        /// <param name="states">形状为 [Batch, 14, 10, 9] 的棋盘状态</param>
        /// <param name="targetPolicies">搜索产生的走法分布 (Pi)</param>
        /// <param name="targetValues">游戏最终胜负结果 (Z)</param>
        public double TrainStep(Tensor states, Tensor targetPolicies, Tensor targetValues)
        {
            _model.train(); // 切换到训练模式，启用 BatchNorm
            _optimizer.zero_grad();

            // 修复 CS1061：TorchSharp 获取维度的方法是 dim() 而非 dimensions
            // 如果收到的不是 4D 张量（即没有 Batch 维度），强制增加 Batch 维度以适配 Conv2d
            if (states.dim() == 3)
            {
                states = states.unsqueeze(0);
            }

            // 1. 前向传播
            var (policyLogits, valuePred) = _model.forward(states);

            // 2. 计算损失函数
            // Value Loss: 均方误差 (MSE)，确保目标值形状为 [Batch, 1]
            var vLoss = torch.nn.functional.mse_loss(valuePred, targetValues.view(-1, 1));

            // Policy Loss: 交叉熵
            var pLoss = ComputePolicyLoss(policyLogits, targetPolicies);

            var totalLoss = vLoss + pLoss;

            // 3. 反向传播与优化
            totalLoss.backward();
            _optimizer.step();

            return totalLoss.item<float>();
        }

        private Tensor ComputePolicyLoss(Tensor logits, Tensor targets)
        {
            // AlphaZero 使用的是预测分布与搜索分布之间的负对数似然
            var logProbs = torch.nn.functional.log_softmax(logits, 1);
            return -(targets * logProbs).sum(1).mean();
        }

        /// <summary>
        /// 调整学习率
        /// </summary>
        public void SetLearningRate(double lr)
        {
            foreach (var group in _optimizer.ParamGroups)
            {
                group.LearningRate = lr;
            }
        }
    }
}