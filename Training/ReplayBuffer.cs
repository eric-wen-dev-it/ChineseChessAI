using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.Training
{
    /// <summary>
    /// 经验回放池：存储自我对弈数据并提供随机采样功能
    /// </summary>
    public class ReplayBuffer
    {
        private readonly int _capacity;
        private readonly List<TrainingExample> _buffer;
        private readonly Random _random = new Random();

        public ReplayBuffer(int capacity = 100000)
        {
            _capacity = capacity;
            _buffer = new List<TrainingExample>(capacity);
        }

        /// <summary>
        /// 添加一局游戏的训练数据
        /// </summary>
        public void AddExamples(List<TrainingExample> examples)
        {
            foreach (var example in examples)
            {
                if (_buffer.Count >= _capacity)
                {
                    // 如果满了，移除最早的经验 (FIFO)
                    _buffer.RemoveAt(0);
                }
                _buffer.Add(example);
            }
            Console.WriteLine($"[ReplayBuffer] 当前缓存条数: {_buffer.Count}");
        }

        /// <summary>
        /// 随机采样一个 Batch 用于训练
        /// </summary>
        public (Tensor States, Tensor Policies, Tensor Values) Sample(int batchSize)
        {
            int count = Math.Min(batchSize, _buffer.Count);

            var batchStates = new List<Tensor>();
            var batchPolicies = new List<Tensor>();
            var batchValues = new List<float>();

            for (int i = 0; i < count; i++)
            {
                int index = _random.Next(_buffer.Count);
                var example = _buffer[index];

                batchStates.Add(example.State);
                batchPolicies.Add(example.Policy);
                batchValues.Add(example.Value);
            }

            // 将 List 转换为单个大 Tensor，并移动到 GPU
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

        public int Count => _buffer.Count;
    }
}