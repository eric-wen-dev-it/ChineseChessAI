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

        public event Action<string>? OnSaveError;

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
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string filePath = Path.Combine(_dataDir, $"game_{timestamp}_{Guid.NewGuid():N}.json");
                string json = JsonSerializer.Serialize(examples);
                File.WriteAllText(filePath, json);
            }
            catch (Exception ex) 
            {
                string msg = $"[ReplayBuffer] 磁盘写入失败: {ex.Message}";
                Console.WriteLine(msg);
                OnSaveError?.Invoke(msg);
            }
        }

        public async Task<(int samples, int games)> LoadOldSamplesAsync(int maxFiles = 200, bool randomize = false, Action<string>? logAction = null, Action<List<Move>, Move, string>? onAuditFailure = null, CancellationToken cancellationToken = default, DateTime? cutoffTime = null)
        {
            if (!Directory.Exists(_dataDir)) return (0, 0);

            var allFilesInfo = Directory.GetFiles(_dataDir, "*.json").Select(f => new FileInfo(f));
            if (cutoffTime.HasValue)
            {
                allFilesInfo = allFilesInfo.Where(f => f.CreationTime < cutoffTime.Value);
            }

            IEnumerable<string> ordered = randomize 
                ? allFilesInfo.OrderBy(_ => _random.Next()).Select(f => f.FullName)
                : allFilesInfo.OrderByDescending(f => f.CreationTime).Select(f => f.FullName);

            var files = ordered.Take(maxFiles).ToList();
            int totalFiles = files.Count;
            int totalLoaded = 0;
            int totalGames = 0;
            int processedCount = 0;

            foreach (var filePath in files)
            {
                if (cancellationToken.IsCancellationRequested) break;
                if (_count >= _capacity) break;
                string fileName = Path.GetFileName(filePath);
                processedCount++;
                
                if (processedCount % 1000 == 0)
                {
                    logAction?.Invoke($"[装载进度] '{Path.GetFileName(_dataDir)}': 已处理 {processedCount}/{totalFiles} 文件, 有效 {totalGames} 局 ({totalLoaded} 条)");
                    await Task.Yield(); // 释放控制权，防止阻塞后台线程导致 UI/其他任务饿死
                }

                try
                {
                    string json = File.ReadAllText(filePath).TrimStart();
                    if (string.IsNullOrEmpty(json)) continue;

                    List<TrainingExample>? examples = null;
                    
                    // --- 1. 探测 JSON 格式 ---
                    bool auditPassed = true;
                    if (json.StartsWith("{")) // 对象格式，可能是 MasterGameData
                    {
                        var masterData = JsonSerializer.Deserialize<MasterGameData>(json);
                        if (masterData != null && masterData.Examples != null)
                        {
                            examples = masterData.Examples;
                            // 【BUG D 修复】：单次反序列化，单次审计
                            if (masterData.MoveHistoryUcci != null && masterData.MoveHistoryUcci.Count > 0)
                            {
                                auditPassed = AuditGame(masterData.MoveHistoryUcci, fileName, logAction, onAuditFailure);
                            }
                        }
                    }
                    else if (json.StartsWith("[")) // 数组格式，直接按列表解析
                    {
                        try { examples = JsonSerializer.Deserialize<List<TrainingExample>>(json); } catch { }
                    }

                    // --- 2. 最终装载数据 ---
                    if (examples != null && examples.Count > 0 && auditPassed)
                    {
                        if (examples[0].State != null && examples[0].State.Length == 14 * 90)
                        {
                            this.AddRange(examples, saveToDisk: false);
                            totalLoaded += examples.Count;
                            totalGames++;
                        }
                    }
                    else if (!auditPassed)
                    {
                        logAction?.Invoke($"[审计拒绝] {fileName}: 存在非法走法，已从训练集中剔除。");
                    }
                }
                catch (Exception ex)
                {
                    logAction?.Invoke($"[装载故障] {fileName}: {ex.Message}");
                }
            }
            return (totalLoaded, totalGames);
        }

        private bool AuditGame(List<string> ucciHistory, string fileName, Action<string>? logAction, Action<List<Move>, Move, string>? onAuditFailure)
        {
            var tempBoard = new Board(); tempBoard.Reset();
            var gen = new Core.MoveGenerator();
            var history = new List<Move>();
            int moveIdx = 1;

            foreach (var ucci in ucciHistory)
            {
                var move = Utils.NotationConverter.UcciToMove(ucci);
                if (!move.HasValue) return false;

                // 审计时允许跳过长打/长捉检测（因为大师对局可能包含此类战术，或者规则集略有不同）
                // 但物理走法和送将必须是合法的。
                string validationResult = gen.GetMoveValidationResult(tempBoard, move.Value, skipPerpetualCheck: true);
                if (validationResult != "合法")
                {
                    sbyte piece = tempBoard.GetPiece(move.Value.From);
                    string pieceName = Board.GetPieceName(piece);
                    string msg = $"第{moveIdx}步 {ucci} 疑似非法 ({pieceName} {validationResult})";
                    
                    logAction?.Invoke($"[审计警告] {fileName} {msg}");
                    onAuditFailure?.Invoke(history, move.Value, msg);
                    return false; // 审计失败
                }

                tempBoard.Push(move.Value.From, move.Value.To);
                history.Add(move.Value);
                moveIdx++;
            }
            return true; // 审计通过
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

        public void Clear()
        {
            lock (_buffer)
            {
                _count = 0;
                _head = 0;
                Array.Clear(_buffer, 0, _capacity);
            }
        }
    }
}