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
            // 核心：强制开启训练模式，重置 BatchNorm 状态
            _model.train();
            _optimizer.zero_grad();

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            // 显式转换并确保不切断梯度链
            using var x = states.to(device).to_type(ScalarType.Float32);
            using var y_policy = targetPolicies.to(device).to_type(ScalarType.Float32);
            using var y_value = targetValues.to(device).to_type(ScalarType.Float32);

            // 前向传播
            var (policyLogits, valuePred) = _model.forward(x);

            // 损失计算：确保操作都在张量上完成
            var vLoss = torch.nn.functional.mse_loss(valuePred, y_value.view(-1, 1));
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(y_policy * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            // 诊断检查
            if (!totalLoss.requires_grad)
            {
                throw new Exception("计算图断裂！建议检查 ResBlock 内部是否包含 RegisterComponents。");
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