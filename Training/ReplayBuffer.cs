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
                    bool isAuditPassed = false;
                    string discardReason = "未知错误";

                    try {
                        var masterData = JsonSerializer.Deserialize<MasterGameData>(json);
                        if (masterData != null && masterData.Examples != null)
                        {
                            if (masterData.Examples.Count <= 10)
                            {
                                discardReason = "对局步数过短 (<10步)";
                            }
                            else if (masterData.MoveHistoryUcci != null && masterData.MoveHistoryUcci.Count > 0)
                            {
                                var tempBoard = new Board(); tempBoard.Reset();
                                var gen = new Core.MoveGenerator();
                                var history = new List<Move>();
                                bool sequenceLegal = true;
                                int moveIdx = 1;

                                foreach (var ucci in masterData.MoveHistoryUcci)
                                {
                                    var move = Utils.NotationConverter.UcciToMove(ucci);
                                    var legalMoves = gen.GenerateLegalMoves(tempBoard);
                                    
                                    if (move == null || !legalMoves.Any(m => m.From == move.Value.From && m.To == move.Value.To))
                                    {
                                        sequenceLegal = false;
                                        sbyte piece = move.HasValue ? tempBoard.GetPiece(move.Value.From) : (sbyte)0;
                                        string pieceName = Board.GetPieceName(piece);
                                        
                                        string detailReason;
                                        if (piece == 0) detailReason = $"第{moveIdx}步 {ucci} 起点无棋子";
                                        else if ((piece > 0) != tempBoard.IsRedTurn) detailReason = $"第{moveIdx}步 {ucci} 走子方错误 (应为{(tempBoard.IsRedTurn?"红":"黑")}方)";
                                        else detailReason = $"第{moveIdx}步 {ucci} 非法 ({pieceName} 违规移动)";
                                        
                                        discardReason = detailReason;
                                        // 触发 UI 演示
                                        if (move.HasValue) onAuditFailure?.Invoke(history, move.Value, detailReason);
                                        break;
                                    }
                                    tempBoard.Push(move.Value.From, move.Value.To);
                                    history.Add(move.Value);
                                    moveIdx++;
                                }
                                if (sequenceLegal) isAuditPassed = true;
                            }
                            else {
                                isAuditPassed = true;
                            }
                            examples = masterData.Examples;
                        }
                    } catch { 
                        discardReason = "JSON 结构解析失败";
                    }

                    if (!isAuditPassed && examples == null) {
                        examples = JsonSerializer.Deserialize<List<TrainingExample>>(json);
                        if (examples != null && examples.Count > 10) isAuditPassed = true;
                    }

                    if (isAuditPassed && examples != null) 
                    {
                        this.AddRange(examples, saveToDisk: false);
                        totalLoaded += examples.Count;
                        totalGames++;
                    }
                    else {
                        logAction?.Invoke($"[审计丢弃] {fileName} 原因: {discardReason}");
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