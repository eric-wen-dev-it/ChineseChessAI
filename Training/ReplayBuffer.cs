using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace ChineseChessAI.Training
{
    public class ReplayBuffer
    {
        private readonly int _capacity;
        private readonly TrainingExample[] _buffer;
        private int _count = 0;
        private int _head = 0;
        private readonly Random _random = new Random();
        private readonly string _dataDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "self_play_data");

        public ReplayBuffer(int capacity = 100000)
        {
            _capacity = capacity;
            _buffer = new TrainingExample[capacity];

            // 确保目录存在
            if (!Directory.Exists(_dataDir))
                Directory.CreateDirectory(_dataDir);
        }

        // --- 保存单局样本到磁盘 ---
        public void SaveExamples(List<TrainingExample> examples)
        {
            try
            {
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                string filePath = Path.Combine(_dataDir, $"game_{timestamp}.json");
                string json = JsonSerializer.Serialize(examples);
                File.WriteAllText(filePath, json);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ReplayBuffer] 保存失败: {ex.Message}");
            }
        }

        // --- 从磁盘加载已有样本 ---
        public void LoadOldSamples()
        {
            if (!Directory.Exists(_dataDir))
                return;

            var files = Directory.GetFiles(_dataDir, "*.json")
                                 .Select(f => new FileInfo(f))
                                 .OrderByDescending(f => f.CreationTime)
                                 .Take(200) // 仅加载最近的 200 局，避免内存撑爆
                                 .ToList();

            int totalLoaded = 0;
            foreach (var file in files)
            {
                try
                {
                    string json = File.ReadAllText(file.FullName);
                    var examples = JsonSerializer.Deserialize<List<TrainingExample>>(json);
                    if (examples != null)
                    {
                        this.AddRange(examples, saveToDisk: false); // 加载时不重复保存
                        totalLoaded += examples.Count;
                    }
                }
                catch { /* 忽略损坏的文件 */ }
            }
            Console.WriteLine($"[ReplayBuffer] 已从磁盘预加载 {totalLoaded} 条样本");
        }

        // 修改 AddRange，增加是否保存的开关
        public void AddRange(List<TrainingExample> newExamples, bool saveToDisk = true)
        {
            if (saveToDisk)
                SaveExamples(newExamples);

            foreach (var ex in newExamples)
            {
                _buffer[_head] = ex;
                _head = (_head + 1) % _capacity;
                if (_count < _capacity)
                    _count++;
            }
        }

        /// <summary>
        /// 【核心修改】：不再直接返回 Tensor 导致爆显存，而是返回原始的 List<TrainingExample>，
        /// 供外部进行 Chunk 切片后再交由 Trainer 转为 Tensor。
        /// </summary>
        public List<TrainingExample> Sample(int batchSize)
        {
            int count = Math.Min(batchSize, _count);
            var batch = new List<TrainingExample>(count);

            for (int i = 0; i < count; i++)
            {
                // 随机抽取样本
                int index = _random.Next(_count);
                batch.Add(_buffer[index]);
            }

            return batch;
        }

        public int Count => _count;
    }
}