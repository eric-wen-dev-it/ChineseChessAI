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

            // --- A. 核心修复：确保输入在正确的设备上并开启梯度追踪 ---
            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            // 确保数据类型为 Float32 且在同一设备
            // 显式先转设备，再转类型
            using var x = states.to(device).to_type(ScalarType.Float32);
            using var y_policy = targetPolicies.to(device).to_type(ScalarType.Float32);
            using var y_value = targetValues.to(device).to_type(ScalarType.Float32);

            // --- B. 前向传播 ---
            var (policyLogits, valuePred) = _model.forward(x);

            // --- C. 损失计算 (必须保持张量计算，不能转回数组) ---
            // 价值损失
            var vLoss = torch.nn.functional.mse_loss(valuePred, y_value.view(-1, 1));

            // 策略损失 (手动实现交叉熵以确保稳定性)
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(y_policy * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            // --- D. 诊断 ---
            if (!totalLoss.requires_grad)
            {
                // 这是一个极端情况：如果这里还是断了，说明 CChessNet 的 forward 内部有问题
                throw new Exception("计算图在 forward 阶段断裂。请检查 CChessNet 的层是否正确 RegisterModule。");
            }

            // --- E. 反向传播与更新 ---
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