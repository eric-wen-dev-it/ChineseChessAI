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

        public (int samples, int games) LoadOldSamples(int maxFiles = 200, bool randomize = false)
        {
            if (!Directory.Exists(_dataDir)) return (0, 0);

            var allPaths = Directory.GetFiles(_dataDir, "*.json");
            IEnumerable<string> ordered = randomize 
                ? allPaths.OrderBy(_ => _random.Next())
                : allPaths.Select(f => new FileInfo(f)).OrderByDescending(f => f.CreationTime).Select(f => f.FullName);

            var files = ordered.Take(maxFiles).ToList();
            int totalLoaded = 0;
            int totalGames = 0;

            foreach (var filePath in files)
            {
                if (_count >= _capacity) break;
                try
                {
                    string json = File.ReadAllText(filePath);
                    List<TrainingExample>? examples = null;

                    // 尝试两种反序列化路径
                    try {
                        var masterData = JsonSerializer.Deserialize<MasterGameData>(json);
                        if (masterData != null && masterData.Examples != null) examples = masterData.Examples;
                    } catch { }

                    if (examples == null) {
                        examples = JsonSerializer.Deserialize<List<TrainingExample>>(json);
                    }

                    if (examples != null && examples.Count > 0) {
                        this.AddRange(examples, saveToDisk: false);
                        totalLoaded += examples.Count;
                        totalGames++;
                    }
                }
                catch { }
            }
            return (totalLoaded, totalGames);
        }

        public void AddRange(List<TrainingExample> newExamples, bool saveToDisk = true)
        {
            if (saveToDisk) SaveExamples(newExamples);
            lock (_buffer) {
                foreach (var ex in newExamples) {
                    _buffer[_head] = ex;
                    _head = (_head + 1) % _capacity;
                    if (_count < _capacity) _count++;
                }
            }
        }

        public List<TrainingExample> Sample(int batchSize)
        {
            lock (_buffer) {
                int count = Math.Min(batchSize, _count);
                var batch = new List<TrainingExample>(count);
                int[] indices = Enumerable.Range(0, _count).OrderBy(_ => _random.Next()).Take(count).ToArray();
                foreach (var i in indices) batch.Add(_buffer[i]);
                return batch;
            }
        }

        public int Count => _count;
    }
}