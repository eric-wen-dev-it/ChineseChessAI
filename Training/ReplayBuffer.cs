using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.Training
{
    public class ReplayBuffer
    {
        private readonly int _capacity;
        private readonly TrainingExample[] _buffer;
        private int _count = 0;
        private int _head = 0;
        private readonly Random _random = new Random();

        public ReplayBuffer(int capacity = 100000)
        {
            _capacity = capacity;
            _buffer = new TrainingExample[capacity];
        }

        // 【修复】重命名为 AddRange 以匹配 MainWindow 的调用
        public void AddRange(List<TrainingExample> newExamples)
        {
            foreach (var ex in newExamples)
            {
                _buffer[_head] = ex;
                _head = (_head + 1) % _capacity;

                if (_count < _capacity)
                    _count++;
            }
        }

        public (Tensor States, Tensor Policies, Tensor Values) Sample(int batchSize)
        {
            int count = Math.Min(batchSize, _count);
            var batchStates = new List<Tensor>();
            var batchPolicies = new List<Tensor>();
            var batchValues = new List<float>();

            for (int i = 0; i < count; i++)
            {
                int index = _random.Next(_count);
                var example = _buffer[index];

                batchStates.Add(torch.tensor(example.State, new long[] { 14, 10, 9 }));
                batchPolicies.Add(torch.tensor(example.Policy, new long[] { 8100 }));
                batchValues.Add(example.Value);
            }

            var states = torch.stack(batchStates);
            var policies = torch.stack(batchPolicies);
            var values = torch.tensor(batchValues.ToArray(), ScalarType.Float32);

            if (cuda.is_available())
            {
                states = states.to(DeviceType.CUDA);
                policies = policies.to(DeviceType.CUDA);
                values = values.to(DeviceType.CUDA);
            }

            return (states, policies, values);
        }

        public int Count => _count;
    }
}