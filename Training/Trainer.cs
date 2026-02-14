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

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            // 将输入移动到计算设备
            using var x = states.to(device).to_type(ScalarType.Float32);
            using var y_p = targetPolicies.to(device).to_type(ScalarType.Float32);
            using var y_v = targetValues.to(device).to_type(ScalarType.Float32);

            // 前向传播
            var (pLogits, vPred) = _model.forward(x);

            // 损失函数
            using var vLoss = torch.nn.functional.mse_loss(vPred, y_v.view(-1, 1));
            using var logSoftmax = torch.nn.functional.log_softmax(pLogits, 1);
            using var pLoss = -(y_p * logSoftmax).sum(1).mean();

            using var totalLoss = vLoss + pLoss;

            if (!totalLoss.requires_grad)
            {
                throw new Exception("严重错误：计算图断裂。请确认模型参数已开启 requires_grad。");
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