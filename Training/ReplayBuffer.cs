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
            // 确保采样数量不超过当前缓存总量
            int count = Math.Min(batchSize, _buffer.Count);
            if (count == 0)
                return (null, null, null);

            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;

            // 临时存放转换回来的张量
            var batchStates = new List<Tensor>();
            var batchPolicies = new List<Tensor>();
            var batchValues = new List<float>();

            for (int i = 0; i < count; i++)
            {
                int index = _random.Next(_buffer.Count);
                var example = _buffer[index];

                // 【核心修正】由于 TrainingExample.State 现在是 float[]
                // 必须通过 from_array 转回 Tensor 才能进行 stack
                batchStates.Add(torch.from_array(example.State).to(device));
                batchPolicies.Add(torch.from_array(example.Policy).to(device));
                batchValues.Add(example.Value);
            }

            // 将所有单条 Tensor 合并为一个批次 Tensor
            var states = torch.stack(batchStates);
            var policies = torch.stack(batchPolicies);
            var values = torch.tensor(batchValues.ToArray(), ScalarType.Float32).to(device);

            return (states, policies, values);
        }

        public int Count => _buffer.Count;
    }
}