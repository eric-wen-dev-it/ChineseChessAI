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
        // 1. 引入梯度缩放器，防止 Float16 下梯度下溢
        private readonly torch.cuda.amp.GradScaler _scaler = new torch.cuda.amp.GradScaler();

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

            if (states.dim() == 3)
                states = states.unsqueeze(0);

            // 2. 开启自动混合精度上下文 (AMP)
            // 这会自动将卷积和全连接层切换到 Tensor Cores (FP16) 计算
            using (var amp = torch.cuda.amp.autocast())
            {
                var (policyLogits, valuePred) = _model.forward(states);

                var vLoss = torch.nn.functional.mse_loss(valuePred, targetValues.view(-1, 1));
                var pLoss = ComputePolicyLoss(policyLogits, targetPolicies);
                var totalLoss = vLoss + pLoss;

                // 3. 使用 Scaler 处理反向传播
                // Scale 损失值 -> Backward -> Step -> Update
                _scaler.scale(totalLoss).backward();
            }

            // 4. 更新权重
            _scaler.step(_optimizer);
            _scaler.update();

            return 0.0; // 返回 Loss 需要从 Tensor 提取，为避免同步阻塞可暂时返回 0 或仅定期提取
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