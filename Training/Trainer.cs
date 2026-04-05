using ChineseChessAI.NeuralNetwork;
using TorchSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace ChineseChessAI.Training
{
    // 【终极防御】：显式定义 struct，确保 System.Text.Json 能 100% 完美保存到硬盘
    public record struct ActionProb(int Index, float Prob);

    // 将 (int Index, float Prob)[] 替换为 ActionProb[]
    public record TrainingExample(float[] State, ActionProb[] SparsePolicy, float Value);

    // 【新增】：大师对局完整数据结构
    public record MasterGameData(List<TrainingExample> Examples, List<string> MoveHistoryUcci);

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

            _optimizer?.Dispose(); // 清理旧的优化器

            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);
            _scheduler = StepLR(_optimizer, step_size: 500, gamma: 0.5);
        }

        // 【核心审计修复】：彻底释放底层 C++ 动量张量池
        public void Dispose()
        {
            _optimizer?.Dispose();
        }

        public float Train(List<TrainingExample> examples, int epochs)
        {
            if (examples == null || examples.Count == 0)
                return 0f;

            using var scope = torch.NewDisposeScope();

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

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

            var statesTensor = stack(statesList).to(device).to_type(ScalarType.Float32);
            var policiesTensor = stack(policiesList).to(device).to_type(ScalarType.Float32);
            var valuesTensor = stack(valuesList).to(device).to_type(ScalarType.Float32);

            double totalLoss = 0;
            for (int e = 0; e < epochs; e++)
            {
                totalLoss += TrainStep(statesTensor, policiesTensor, valuesTensor);
            }

            _iterationCount++;
            _scheduler.step();

            return (float)(totalLoss / epochs);
        }

        private double TrainStep(Tensor statesDevice, Tensor targetPoliciesDevice, Tensor targetValuesDevice)
        {
            _model.train();
            _optimizer.zero_grad();

            var (policyLogits, valuePred) = _model.forward(statesDevice);

            var vLoss = torch.nn.functional.mse_loss(valuePred, targetValuesDevice.view(-1, 1));
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(targetPoliciesDevice * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            if (!totalLoss.requires_grad)
                throw new Exception($"计算图断裂！参数状态: {_model.parameters().First().requires_grad}");

            totalLoss.backward();
            _optimizer.step();

            return totalLoss.item<float>();
        }

        public double GetCurrentLR()
        {
            var groups = _optimizer.ParamGroups;
            return groups != null && groups.Any() ? groups.First().LearningRate : _learningRate;
        }
    }
}