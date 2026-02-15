using ChineseChessAI.NeuralNetwork;
using TorchSharp;
using static TorchSharp.torch;

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
            var parameters = _model.parameters().Where(p => p.requires_grad).ToList();
            if (parameters.Count == 0)
                return;
            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);
        }

        // 【新增】供 MainWindow 调用的多轮训练接口
        public float Train((Tensor States, Tensor Policies, Tensor Values) batch, int epochs)
        {
            double totalLoss = 0;
            for (int e = 0; e < epochs; e++)
            {
                totalLoss += TrainStep(batch.States, batch.Policies, batch.Values);
            }
            return (float)(totalLoss / epochs);
        }

        public double TrainStep(Tensor states, Tensor targetPolicies, Tensor targetValues)
        {
            using var scope = torch.NewDisposeScope();

            _model.train();
            _optimizer.zero_grad();

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            var x = states.to(device).to_type(ScalarType.Float32);
            var y_policy = targetPolicies.to(device).to_type(ScalarType.Float32);
            var y_value = targetValues.to(device).to_type(ScalarType.Float32);

            var (policyLogits, valuePred) = _model.forward(x);

            var vLoss = torch.nn.functional.mse_loss(valuePred, y_value.view(-1, 1));
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(y_policy * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            if (!totalLoss.requires_grad)
            {
                throw new Exception($"计算图断裂！参数状态: {_model.parameters().First().requires_grad}");
            }

            totalLoss.backward();
            _optimizer.step();

            return totalLoss.item<float>();
        }
    }
}