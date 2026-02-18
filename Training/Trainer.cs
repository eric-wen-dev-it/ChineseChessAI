using ChineseChessAI.NeuralNetwork;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler; // 【新增】引用调度器命名空间

namespace ChineseChessAI.Training
{
    public class Trainer
    {
        private readonly CChessNet _model;
        private torch.optim.Optimizer _optimizer;
        private LRScheduler _scheduler; // 【新增】学习率调度器实例
        private readonly double _learningRate = 0.001;

        // 【优化】记录迭代次数，用于触发学习率更新
        private int _iterationCount = 0;

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

            // 1. 初始化 Adam 优化器
            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);

            // 2. 【新增】初始化学习率调度器
            // 每隔 50 次训练调用 (step_size)，学习率变为原来的 50% (gamma)
            // 你可以根据日志观察，如果 Loss 持续不降，可以把 step_size 调小
            _scheduler = StepLR(_optimizer, step_size: 50, gamma: 0.5);
        }

        // 供 MainWindow 调用的多轮训练接口
        public float Train((Tensor States, Tensor Policies, Tensor Values) batch, int epochs)
        {
            double totalLoss = 0;
            for (int e = 0; e < epochs; e++)
            {
                totalLoss += TrainStep(batch.States, batch.Policies, batch.Values);
            }

            // 【关键】每一轮训练（Iteration）结束后，更新学习率调度器状态
            _iterationCount++;
            _scheduler.step();

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

            // Loss 计算逻辑
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

        // 【修正后】获取当前学习率的方法
        public double GetCurrentLR()
        {
            // TorchSharp 中，Adam 优化器的 Options 存储了初始学习率
            // 而实际运行中的学习率需要通过这种方式访问：
            var groups = _optimizer.ParamGroups;
            if (groups != null && groups.Any())
            {
                return groups.First().LearningRate;
            }
            return _learningRate;
        }
    }
}