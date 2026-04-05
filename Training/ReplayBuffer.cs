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
        private readonly string _dataDir;

        public string DataDir => _dataDir;

        public ReplayBuffer(int capacity = 100000, string dataDir = null)
        {
            _capacity = capacity;
            _buffer = new TrainingExample[capacity];
            _dataDir = dataDir ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "self_play_data");
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

        public int LoadOldSamples(int maxFiles = 200, bool randomize = false)
        {
            if (!Directory.Exists(_dataDir))
                return 0;

            var allPaths = Directory.GetFiles(_dataDir, "*.json");

            IEnumerable<string> ordered;
            if (randomize)
            {
                // 随机打散，确保从整个数据集中均匀采样，而非只取最新文件
                ordered = allPaths.OrderBy(_ => _random.Next());
            }
            else
            {
                // 自对弈缓存：取最近 N 个文件（有时效性要求）
                ordered = allPaths.Select(f => new FileInfo(f))
                                  .OrderByDescending(f => f.CreationTime)
                                  .Select(f => f.FullName);
            }

            var files = ordered.Take(maxFiles).ToList();
            string absPath = Path.GetFullPath(_dataDir);
            Console.WriteLine($"[ReplayBuffer] 正在从目录装载数据: {absPath}");

            int totalLoaded = 0;
            foreach (var filePath in files)
            {
                if (_count >= _capacity) break; // 缓冲区满时提前退出，避免无谓 IO
                try
                {
                    string json = File.ReadAllText(filePath);
                    var examples = JsonSerializer.Deserialize<List<TrainingExample>>(json);
                    if (examples != null)
                    {
                        this.AddRange(examples, saveToDisk: false);
                        totalLoaded += examples.Count;
                    }
                }
                catch { /* 忽略损坏的文件 */ }
            }
            Console.WriteLine($"[ReplayBuffer] 已从磁盘预加载 {totalLoaded} 条样本（共 {allPaths.Length} 个文件可用）");
            return totalLoaded;
        }

        public void AddRange(List<TrainingExample> newExamples, bool saveToDisk = true)
        {
            if (saveToDisk)
                SaveExamples(newExamples);

            lock (_buffer) // 【并发保护】：确保多线程写入安全
            {
                foreach (var ex in newExamples)
                {
                    _buffer[_head] = ex;
                    _head = (_head + 1) % _capacity;
                    if (_count < _capacity)
                        _count++;
                }
            }
        }

        public List<TrainingExample> Sample(int batchSize)
        {
            lock (_buffer) // 【并发保护】：防止多线程竞争导致 Random 崩溃
            {
                int count = Math.Min(batchSize, _count);
                var batch = new List<TrainingExample>(count);

                // 使用 Fisher-Yates 思想的快速索引采样
                int[] indices = Enumerable.Range(0, _count).ToArray();
                for (int i = 0; i < count; i++)
                {
                    int j = _random.Next(i, _count);
                    (indices[i], indices[j]) = (indices[j], indices[i]);
                    batch.Add(_buffer[indices[i]]);
                }

                return batch;
            }
        }

        public int Count => _count;
    }
}