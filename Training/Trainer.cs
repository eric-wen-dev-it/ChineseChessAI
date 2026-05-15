using ChineseChessAI.Core;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System.IO;
using System.Text.Json;
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
        private const double InitialLearningRate = 0.0002;
        private const double MinLearningRate = 0.00001;
        private const float HardValueTarget = 0.95f;
        private const float TeacherValueWeight = 0.65f;
        private const float TeacherPolicyWeight = 0.35f;
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

            _optimizer = torch.optim.Adam(parameters, InitialLearningRate, weight_decay: 1e-4);
            _scheduler = StepLR(_optimizer, step_size: 500, gamma: 0.5);
            ClampLearningRate();
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
                    float[] densePolicy = BuildPolicyTarget(ex);

                    policiesList.Add(tensor(densePolicy));
                    float valueTarget = ex.TeacherValue.HasValue
                        ? Lerp(ex.Value, ex.TeacherValue.Value, TeacherValueWeight)
                        : ex.Value;
                    valuesList.Add(tensor(SmoothValueTarget(valueTarget)));
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
                ClampLearningRate();
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

        private static float SmoothValueTarget(float value)
        {
            if (value >= 1.0f)
                return HardValueTarget;
            if (value <= -1.0f)
                return -HardValueTarget;
            return value;
        }

        private static float[] BuildPolicyTarget(TrainingExample ex)
        {
            float[] densePolicy = new float[8100];
            AddSparsePolicy(densePolicy, ex.SparsePolicy, 1.0f);

            if (ex.TeacherSparsePolicy is { Length: > 0 })
            {
                for (int i = 0; i < densePolicy.Length; i++)
                    densePolicy[i] *= 1.0f - TeacherPolicyWeight;

                AddSparsePolicy(densePolicy, ex.TeacherSparsePolicy, TeacherPolicyWeight);
            }

            NormalizePolicy(densePolicy);
            return densePolicy;
        }

        private static void AddSparsePolicy(float[] densePolicy, ActionProb[]? sparsePolicy, float weight)
        {
            if (sparsePolicy == null)
                return;

            foreach (var p in sparsePolicy)
            {
                if (p.Index >= 0 && p.Index < densePolicy.Length && float.IsFinite(p.Prob) && p.Prob > 0)
                    densePolicy[p.Index] += p.Prob * weight;
            }
        }

        private static void NormalizePolicy(float[] densePolicy)
        {
            float sum = 0f;
            for (int i = 0; i < densePolicy.Length; i++)
                sum += densePolicy[i];

            if (sum <= 0 || !float.IsFinite(sum))
                return;

            for (int i = 0; i < densePolicy.Length; i++)
                densePolicy[i] /= sum;
        }

        private static float Lerp(float from, float to, float weight)
        {
            return from + (to - from) * Math.Clamp(weight, 0.0f, 1.0f);
        }

        public double GetCurrentLR()
        {
            var groups = _optimizer.ParamGroups;
            return groups != null && groups.Any() ? groups.First().LearningRate : InitialLearningRate;
        }

        private void ClampLearningRate()
        {
            var groups = _optimizer.ParamGroups;
            if (groups == null)
                return;

            foreach (var group in groups)
            {
                if (group.LearningRate < MinLearningRate)
                    group.LearningRate = MinLearningRate;
            }
        }

        // 持久化 Adam 动量缓冲区 + 调度器步数，避免 agent 卸载/重载时退化为"全新优化器"。
        // optimizerPath: TorchSharp 二进制；sidecarPath: JSON，存 _iterationCount。
        public void SaveOptimizerState(string optimizerPath, string sidecarPath)
        {
            try
            {
                string? dir = Path.GetDirectoryName(optimizerPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                ((TorchSharp.Modules.OptimizerHelper)_optimizer).save_state_dict(optimizerPath);
                File.WriteAllText(sidecarPath, JsonSerializer.Serialize(new OptimizerSidecar
                {
                    IterationCount = _iterationCount,
                    LearningRate = GetCurrentLR()
                }));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Trainer] 优化器状态保存失败: {ex.Message}");
            }
        }

        // 调用前提：调用方已经先 model.load() 完毕、CompleteInit() 也已执行（即 _optimizer/_scheduler 已重建）。
        // 文件不存在时静默 no-op，等价于"全新优化器"——保留之前的行为。
        public bool TryLoadOptimizerState(string optimizerPath, string sidecarPath)
        {
            if (!File.Exists(optimizerPath))
                return false;

            try
            {
                ((TorchSharp.Modules.OptimizerHelper)_optimizer).load_state_dict(optimizerPath);

                int restoredIterations = 0;
                if (File.Exists(sidecarPath))
                {
                    string json = File.ReadAllText(sidecarPath);
                    var sidecar = JsonSerializer.Deserialize<OptimizerSidecar>(json);
                    if (sidecar != null)
                        restoredIterations = sidecar.IterationCount;
                }

                _iterationCount = restoredIterations;
                // StepLR 没有 last_epoch 直接 setter；用 step() 推进到对应位置。
                // step_size=500，所以这是 O(restoredIterations) 但每次只读一个 int + 调一次 LR 更新，毫秒级。
                for (int i = 0; i < restoredIterations; i++)
                {
                    _scheduler.step();
                    ClampLearningRate();
                }

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Trainer] 优化器状态加载失败，回退为全新优化器: {ex.Message}");
                // 失败时把已经被部分 load 的优化器清理重建，避免半 load 状态。
                ResetOptimizer();
                _iterationCount = 0;
                return false;
            }
        }

        private sealed class OptimizerSidecar
        {
            public int IterationCount { get; set; }
            public double LearningRate { get; set; }
        }
    }
}
