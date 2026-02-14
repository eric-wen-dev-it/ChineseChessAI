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

            // 修复：不要对 input 使用 using，因为它们在 MainWindow 的 DisposeScope 中被管理
            var x = states.to(device).to_type(ScalarType.Float32);
            var y_policy = targetPolicies.to(device).to_type(ScalarType.Float32);
            var y_value = targetValues.to(device).to_type(ScalarType.Float32);

            // 执行推理
            var (policyLogits, valuePred) = _model.forward(x);

            // 损失计算
            var vLoss = torch.nn.functional.mse_loss(valuePred, y_value.view(-1, 1));
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(y_policy * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            if (!totalLoss.requires_grad)
            {
                throw new Exception("计算图断裂！此时请务必确认 MainWindow 中的 parameters 循环是否执行成功。");
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