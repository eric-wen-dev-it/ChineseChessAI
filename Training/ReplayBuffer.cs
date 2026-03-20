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
            if (!Directory.Exists(_dataDir))
                Directory.CreateDirectory(_dataDir);
        }

        public void SaveExamples(List<TrainingExample> examples)
        {
            try
            {
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                string filePath = Path.Combine(_dataDir, $"game_{timestamp}.json");
                string json = JsonSerializer.Serialize(examples);
                File.WriteAllText(filePath, json);
            }
            catch (Exception ex) { Console.WriteLine($"[ReplayBuffer] 保存失败: {ex.Message}"); }
        }

        public void LoadOldSamples()
        {
            if (!Directory.Exists(_dataDir))
                return;

            var files = Directory.GetFiles(_dataDir, "*.json")
                                 .Select(f => new FileInfo(f))
                                 .OrderByDescending(f => f.CreationTime)
                                 .Take(200)
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
                        this.AddRange(examples, saveToDisk: false);
                        totalLoaded += examples.Count;
                    }
                }
                catch { /* 忽略损坏的文件 */ }
            }
            Console.WriteLine($"[ReplayBuffer] 已从磁盘预加载 {totalLoaded} 条样本");
        }

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

        public List<TrainingExample> Sample(int batchSize)
        {
            int count = Math.Min(batchSize, _count);
            var batch = new List<TrainingExample>(count);

            // 【核心修复】：无放回采样 (Fisher-Yates 洗牌算法变体)，避免过度拟合同一状态
            var indices = Enumerable.Range(0, _count).OrderBy(x => _random.Next()).Take(count);
            foreach (var index in indices)
            {
                batch.Add(_buffer[index]);
            }

            return batch;
        }

        public int Count => _count;
    }
}