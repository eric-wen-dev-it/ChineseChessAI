using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using ChineseChessAI.Core;

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

        public (int samples, int games) LoadOldSamples(int maxFiles = 200, bool randomize = false, Action<string>? logAction = null, Action<List<Move>, Move, string>? onAuditFailure = null)
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
                string fileName = Path.GetFileName(filePath);
                try
                {
                    string json = File.ReadAllText(filePath);
                    List<TrainingExample>? examples = null;
                    
                    // --- 1. 尝试解析大师数据格式 ---
                    MasterGameData? masterData = null;
                    try { masterData = JsonSerializer.Deserialize<MasterGameData>(json); } catch { }

                    if (masterData != null && masterData.Examples != null && masterData.Examples.Count > 0)
                    {
                        examples = masterData.Examples;
                        
                        // 执行审计（但不因审计失败而丢弃数据，仅用于演示和日志）
                        if (masterData.MoveHistoryUcci != null && masterData.MoveHistoryUcci.Count > 0)
                        {
                            AuditGame(masterData.MoveHistoryUcci, fileName, logAction, onAuditFailure);
                        }
                    }
                    else
                    {
                        // --- 2. 尝试解析纯样本数组格式 (Fallback) ---
                        try { examples = JsonSerializer.Deserialize<List<TrainingExample>>(json); } catch { }
                    }

                    // --- 3. 最终装载数据 ---
                    if (examples != null && examples.Count > 0)
                    {
                        // 确保数据维度正确
                        if (examples[0].State != null && examples[0].State.Length == 14 * 90)
                        {
                            this.AddRange(examples, saveToDisk: false);
                            totalLoaded += examples.Count;
                            totalGames++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    logAction?.Invoke($"[装载故障] {fileName}: {ex.Message}");
                }
            }
            return (totalLoaded, totalGames);
        }

        private void AuditGame(List<string> ucciHistory, string fileName, Action<string>? logAction, Action<List<Move>, Move, string>? onAuditFailure)
        {
            var tempBoard = new Board(); tempBoard.Reset();
            var gen = new Core.MoveGenerator();
            var history = new List<Move>();
            int moveIdx = 1;

            foreach (var ucci in ucciHistory)
            {
                var move = Utils.NotationConverter.UcciToMove(ucci);
                if (!move.HasValue) break;

                string validationResult = gen.GetMoveValidationResult(tempBoard, move.Value);
                if (validationResult != "合法")
                {
                    sbyte piece = tempBoard.GetPiece(move.Value.From);
                    string pieceName = Board.GetPieceName(piece);
                    string msg = $"第{moveIdx}步 {ucci} 疑似非法 ({pieceName} {validationResult})";
                    
                    logAction?.Invoke($"[审计警告] {fileName} {msg}");
                    onAuditFailure?.Invoke(history, move.Value, msg);
                    break; // 发现一处错误即停止后续审计
                }

                tempBoard.Push(move.Value.From, move.Value.To);
                history.Add(move.Value);
                moveIdx++;
            }
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