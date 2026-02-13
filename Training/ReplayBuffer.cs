using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.Training
{
    public class ReplayBuffer
    {
        private readonly int _capacity;
        // 修复：使用定长数组而非 List，避免 RemoveAt(0) 的巨大开销
        private readonly TrainingExample[] _buffer;
        private int _count = 0;
        private int _head = 0; // 循环指针
        private readonly Random _random = new Random();

        public ReplayBuffer(int capacity = 100000)
        {
            _capacity = capacity;
            _buffer = new TrainingExample[capacity];
        }

        public void AddExamples(List<TrainingExample> newExamples)
        {
            foreach (var ex in newExamples)
            {
                // 使用循环数组逻辑：O(1) 复杂度覆盖旧数据
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