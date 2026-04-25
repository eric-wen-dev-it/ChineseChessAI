using ChineseChessAI.Core;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace ChineseChessAI.Training
{
    public class Trainer : IDisposable
    {
        private readonly CChessNet _model;
        private torch.optim.Optimizer _optimizer;
        private LRScheduler _scheduler;
        private readonly double _learningRate = 0.0002;
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

            (_scheduler as IDisposable)?.Dispose(); // 安全清理旧的调度器
            _optimizer?.Dispose(); // 清理旧的优化器

            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);
            _scheduler = StepLR(_optimizer, step_size: 500, gamma: 0.5);
        }

        // 【核心审计修复】：彻底释放底层 C++ 动量张量池
        public void Dispose()
        {
            (_scheduler as IDisposable)?.Dispose();
            _optimizer?.Dispose();
        }

        public float Train(List<TrainingExample> examples, int epochs)
        {
            if (examples == null || examples.Count == 0)
                return 0f;

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            // 【关键修复】：用窄作用域只覆盖输入张量构建阶段。
            // 不能用单一大 DisposeScope 包裹整个 Train()，否则 Adam 在首次 step() 时
            // 创建的动量缓冲区（exp_avg / exp_avg_sq）会被注册到 scope，Train 退出时
            // scope.Dispose() 会释放这些缓冲区，下一轮调用 step() 时
            // handle 已为 IntPtr.Zero → "Tensor invalid -- empty handle"。
            Tensor statesTensor, policiesTensor, valuesTensor;
            using (var buildScope = torch.NewDisposeScope())
            {
                var statesList = new List<Tensor>(examples.Count);
                var policiesList = new List<Tensor>(examples.Count);
                var valuesList = new List<Tensor>(examples.Count);

                foreach (var ex in examples)
                {
                    statesList.Add(tensor(ex.State).view(14, 10, 9));

                    // 将节约内存的稀疏策略，在训练这一刻还原为稠密 Tensor (用完即抛，不占常驻内存)
                    float[] densePolicy = new float[8100];
                    foreach (var p in ex.SparsePolicy)
                        densePolicy[p.Index] = p.Prob;

                    policiesList.Add(tensor(densePolicy));
                    valuesList.Add(tensor(ex.Value));
                }

                // MoveToOuterDisposeScope：将 stacked 张量移出 buildScope，
                // 使其成为手动管理张量（无 scope 追踪），由下方 finally 负责释放。
                statesTensor = stack(statesList).to(device).to_type(ScalarType.Float32).MoveToOuterDisposeScope();
                policiesTensor = stack(policiesList).to(device).to_type(ScalarType.Float32).MoveToOuterDisposeScope();
                valuesTensor = stack(valuesList).to(device).to_type(ScalarType.Float32).MoveToOuterDisposeScope();
            }

            double totalLoss = 0;
            try
            {
                for (int e = 0; e < epochs; e++)
                {
                    totalLoss += TrainStep(statesTensor, policiesTensor, valuesTensor);
                }

                _iterationCount++;
                _scheduler.step();
            }
            finally
            {
                statesTensor.Dispose();
                policiesTensor.Dispose();
                valuesTensor.Dispose();
            }

            return (float)(totalLoss / epochs);
        }

        private double TrainStep(Tensor statesDevice, Tensor targetPoliciesDevice, Tensor targetValuesDevice)
        {
            return GpuExecutionGate.Run(() =>
            {
                _model.train();
                _optimizer.zero_grad();

                // 【关键修复】：DisposeScope 仅覆盖 forward + backward（激活张量），
                // _optimizer.step() 在 scope 退出后调用。
                // Adam 动量缓冲区在 step() 内创建时没有活跃 scope，
                // 因此不会被注册到任何 scope，可以在 Trainer 生命周期内持久存在。
                float lossValue;
                using (var forwardScope = torch.NewDisposeScope())
                {
                    var (policyLogits, valuePred) = _model.forward(statesDevice);

                    var vLoss = torch.nn.functional.mse_loss(valuePred, targetValuesDevice.view(-1, 1));
                    var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
                    var pLoss = -(targetPoliciesDevice * logProbs).sum(1).mean();

                    var totalLoss = vLoss + pLoss;

                    if (!totalLoss.requires_grad)
                        throw new Exception($"计算图断裂！参数状态: {_model.parameters().First().requires_grad}");

                    totalLoss.backward();
                    lossValue = totalLoss.item<float>();
                    // forwardScope 退出：policyLogits、valuePred、vLoss、logProbs、pLoss、totalLoss 全部释放
                    // 参数梯度（param.grad）附属于模型参数，不受 scope 影响，optimizer.step() 可正常读取
                }

                _optimizer.step(); // 在 scope 外调用：动量张量不被任何 scope 捕获，生命周期与 Trainer 绑定
                return (double)lossValue;
            });
        }

        public double GetCurrentLR()
        {
            var groups = _optimizer.ParamGroups;
            return groups != null && groups.Any() ? groups.First().LearningRate : _learningRate;
        }
    }
}
