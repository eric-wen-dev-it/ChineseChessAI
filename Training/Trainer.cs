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
        private readonly double _l2Reg = 1e-4;

        public Trainer(CChessNet model)
        {
            _model = model;
            _optimizer = torch.optim.Adam(_model.parameters(), _learningRate, weight_decay: _l2Reg);
        }

        public double TrainStep(Tensor states, Tensor targetPolicies, Tensor targetValues)
        {
            _model.train();
            _optimizer.zero_grad();

            // 维度保护：确保是 4D 张量 [Batch, Channels, Height, Width]
            if (states.dim() == 3)
                states = states.unsqueeze(0);

            // --- 标准 FP32 训练 (移除 AMP) ---

            // 1. 前向传播
            var (policyLogits, valuePred) = _model.forward(states);

            // 2. 计算损失
            var vLoss = torch.nn.functional.mse_loss(valuePred, targetValues.view(-1, 1));
            var pLoss = ComputePolicyLoss(policyLogits, targetPolicies);
            var totalLoss = vLoss + pLoss;

            // 3. 反向传播
            totalLoss.backward();
            _optimizer.step();

            return totalLoss.item<float>();
        }

        private Tensor ComputePolicyLoss(Tensor logits, Tensor targets)
        {
            var logProbs = torch.nn.functional.log_softmax(logits, 1);
            return -(targets * logProbs).sum(1).mean();
        }

        public void SetLearningRate(double lr)
        {
            foreach (var group in _optimizer.ParamGroups)
                group.LearningRate = lr;
        }
    }
}