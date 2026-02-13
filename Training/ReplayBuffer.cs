using System;
using System.Collections.Generic;
using System.Linq; // 必须添加，用于 Select
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
        // 你代码中定义的是 _buffer，所以 AddExamples 里也必须用 _buffer
        private readonly List<TrainingExample> _buffer;
        private readonly Random _random = new Random();

        public ReplayBuffer(int capacity = 100000)
        {
            _capacity = capacity;
            _buffer = new List<TrainingExample>(capacity);
        }

        /// <summary>
        /// 直接接收转换好的 TrainingExample (数组形式) 列表
        /// </summary>
        public void AddExamples(List<TrainingExample> newExamples)
        {
            foreach (var ex in newExamples)
            {
                // 修正变量名：将 _examples 改为类定义的 _buffer
                if (this._buffer.Count >= _capacity)
                {
                    this._buffer.RemoveAt(0);
                }
                this._buffer.Add(ex);
            }
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

                // 核心修正：必须指定原始数据的形状 [14, 10, 9]
                batchStates.Add(torch.tensor(example.State, new long[] { 14, 10, 9 }));
                batchPolicies.Add(torch.tensor(example.Policy, new long[] { 8100 }));
                batchValues.Add(example.Value);
            }

            // stack 后 states 形状变为 [Batch, 14, 10, 9]
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