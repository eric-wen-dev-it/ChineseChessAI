using ChineseChessAI.NeuralNetwork;
using TorchSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler;

namespace ChineseChessAI.Training
{
    public class Trainer
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

            _optimizer = torch.optim.Adam(parameters, _learningRate, weight_decay: 1e-4);
            _scheduler = StepLR(_optimizer, step_size: 500, gamma: 0.5);
        }

        public float Train(List<TrainingExample> examples, int epochs)
        {
            if (examples == null || examples.Count == 0)
                return 0f;

            var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            var statesList = new List<Tensor>(examples.Count);
            var policiesList = new List<Tensor>(examples.Count);
            var valuesList = new List<Tensor>(examples.Count);

            foreach (var ex in examples)
            {
                statesList.Add(tensor(ex.State).view(14, 10, 9));
                policiesList.Add(tensor(ex.Policy));
                valuesList.Add(tensor(ex.Value));
            }

            // 【核心提速修复】：在 Epoch 循环外，一次性把所有数据搬进显卡 (to(device))
            using var statesTensor = stack(statesList).to(device).to_type(ScalarType.Float32);
            using var policiesTensor = stack(policiesList).to(device).to_type(ScalarType.Float32);
            using var valuesTensor = stack(valuesList).to(device).to_type(ScalarType.Float32);

            double totalLoss = 0;

            for (int e = 0; e < epochs; e++)
            {
                // 传入的已经是显卡上的 Tensor，TrainStep 内部不再搬运
                totalLoss += TrainStep(statesTensor, policiesTensor, valuesTensor);
            }

            foreach (var t in statesList)
                t.Dispose();
            foreach (var t in policiesList)
                t.Dispose();
            foreach (var t in valuesList)
                t.Dispose();

            _iterationCount++;
            _scheduler.step();

            return (float)(totalLoss / epochs);
        }

        private double TrainStep(Tensor statesDevice, Tensor targetPoliciesDevice, Tensor targetValuesDevice)
        {
            using var scope = torch.NewDisposeScope();

            _model.train();
            _optimizer.zero_grad();

            // 【性能解放】：直接使用传入的显存 Tensor，剔除了耗时的 .to(device) 操作
            var (policyLogits, valuePred) = _model.forward(statesDevice);

            var vLoss = torch.nn.functional.mse_loss(valuePred, targetValuesDevice.view(-1, 1));
            var logProbs = torch.nn.functional.log_softmax(policyLogits, 1);
            var pLoss = -(targetPoliciesDevice * logProbs).sum(1).mean();

            var totalLoss = vLoss + pLoss;

            if (!totalLoss.requires_grad)
            {
                throw new Exception($"计算图断裂！参数状态: {_model.parameters().First().requires_grad}");
            }

            totalLoss.backward();
            _optimizer.step();

            return totalLoss.item<float>();
        }

        public double GetCurrentLR()
        {
            var groups = _optimizer.ParamGroups;
            if (groups != null && groups.Any())
            {
                return groups.First().LearningRate;
            }
            return _learningRate;
        }
    }
}