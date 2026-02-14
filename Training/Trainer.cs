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
            ResetOptimizer();
        }

        public void ResetOptimizer()
        {
            // 确保只优化 requires_grad = true 的参数
            var parameters = _model.parameters().Where(p => p.requires_grad).ToList();
            if (parameters.Count == 0)
                return;
            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);
        }

        public double TrainStep(Tensor states, Tensor targetPolicies, Tensor targetValues)
        {
            // 【关键修复】添加 Scope，自动回收本轮训练产生的所有中间张量（防止显存撑爆卡死）
            using var scope = torch.NewDisposeScope();

            _model.train();
            _optimizer.zero_grad();

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            // 1. 数据转移（由 scope 管理生命周期，无需手动 using）
            var x = states.to(device).to_type(ScalarType.Float32);
            var y_policy = targetPolicies.to(device).to_type(ScalarType.Float32);
            var y_value = targetValues.to(device).to_type(ScalarType.Float32);

            // 2. 前向传播 (此时内部没有 using，梯度链完好)
            var (policyLogits, valuePred) = _model.forward(x);

            // 3. 损失计算
            var vLoss = torch.nn.functional.mse_loss(valuePred, y_value.view(-1, 1));
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(y_policy * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            // 4. 诊断：此时如果还断裂，那就是模型结构问题（但您已修复结构，所以这里应该通过）
            if (!totalLoss.requires_grad)
            {
                throw new Exception($"计算图断裂！参数状态: {_model.parameters().First().requires_grad}");
            }

            // 5. 反向传播 & 更新
            totalLoss.backward();
            _optimizer.step();

            // 返回 Loss 数值 (item<float> 会将数据拷回 CPU，是安全的)
            return totalLoss.item<float>();
        } // <--- scope 结束，x, y, policyLogits, totalLoss 以及 ResBlock 中间变量全部释放
    }
}