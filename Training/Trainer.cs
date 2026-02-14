using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using ChineseChessAI.NeuralNetwork;

namespace ChineseChessAI.Training
{
    public class Trainer
    {
        private readonly CChessNet _model;
        private torch.optim.Optimizer _optimizer;
        private readonly double _learningRate = 0.001;

        public Trainer(CChessNet model)
        {
            _model = model;
            // 延迟初始化优化器，确保拿到的是最新的参数句柄
            ResetOptimizer();
        }

        public void ResetOptimizer()
        {
            // 核心修复：只给开启了梯度的参数创建优化器
            var parameters = _model.parameters().Where(p => p.requires_grad).ToList();
            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);
        }

        public double TrainStep(Tensor states, Tensor targetPolicies, Tensor targetValues)
        {
            _model.train();
            _optimizer.zero_grad();

            // 1. 前向传播
            var (policyLogits, valuePred) = _model.forward(states);

            // 2. 计算损失
            var vLoss = torch.nn.functional.mse_loss(valuePred, targetValues.view(-1, 1));
            var pLoss = ComputePolicyLoss(policyLogits, targetPolicies);
            var totalLoss = vLoss + pLoss;

            // 3. 反向传播
            // 确保 totalLoss 本身需要梯度
            if (!totalLoss.requires_grad)
            {
                throw new Exception("Loss tensor does not require grad. 计算图断裂！");
            }

            totalLoss.backward();
            _optimizer.step();

            return totalLoss.item<float>();
        }

        private Tensor ComputePolicyLoss(Tensor logits, Tensor targets)
        {
            var logProbs = torch.nn.functional.log_softmax(logits, 1);
            return -(targets * logProbs).sum(1).mean();
        }
    }
}