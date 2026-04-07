using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using ChineseChessAI.Core;

namespace ChineseChessAI.Training
{
    public class ReplayBuffer
    {
        private readonly int _capacity;
        private readonly TrainingExample[] _buffer;
        private int _count;
        private int _head;
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

        public async Task<(int samples, int games)> LoadOldSamplesAsync(
            int maxFiles = 200,
            bool randomize = false,
            Action<string>? logAction = null,
            Action<List<Move>, Move, string>? onAuditFailure = null,
            CancellationToken cancellationToken = default,
            DateTime? cutoffTime = null)
        {
            if (!Directory.Exists(_dataDir))
                return (0, 0);

            var allFilesInfo = Directory.GetFiles(_dataDir, "*.json").Select(f => new FileInfo(f));
            if (cutoffTime.HasValue)
                allFilesInfo = allFilesInfo.Where(f => f.CreationTime < cutoffTime.Value);

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
                if (cancellationToken.IsCancellationRequested)
                    break;
                if (_count >= _capacity)
                    break;

                string fileName = Path.GetFileName(filePath);
                processedCount++;

                if (processedCount % 1000 == 0)
                {
                    logAction?.Invoke($"[装载进度] '{Path.GetFileName(_dataDir)}': 已处理 {processedCount}/{totalFiles} 文件, 有效 {totalGames} 局 ({totalLoaded} 条)");
                    await Task.Yield();
                }

                try
                {
                    string json = File.ReadAllText(filePath).TrimStart();
                    if (string.IsNullOrEmpty(json))
                        continue;

                    List<TrainingExample>? examples = null;
                    bool auditPassed = true;

                    if (json.StartsWith("{"))
                    {
                        var masterData = JsonSerializer.Deserialize<MasterGameData>(json);
                        if (masterData != null && masterData.Examples != null)
                        {
                            examples = masterData.Examples;
                            if (masterData.MoveHistoryUcci != null && masterData.MoveHistoryUcci.Count > 0)
                            {
                                auditPassed = AuditGame(masterData.MoveHistoryUcci, fileName, logAction, onAuditFailure);
                            }
                            else
                            {
                                auditPassed = false;
                                logAction?.Invoke($"[审计拒绝] {fileName}: 缺少 MoveHistoryUcci，无法验证数据合法性。");
                            }
                        }
                    }
                    else if (json.StartsWith("["))
                    {
                        try
                        {
                            examples = JsonSerializer.Deserialize<List<TrainingExample>>(json);
                        }
                        catch
                        {
                        }
                    }

                    if (examples != null && examples.Count > 0 && auditPassed)
                    {
                        if (examples[0].State != null && examples[0].State.Length == 14 * 90)
                        {
                            AddRange(examples, saveToDisk: false);
                            totalLoaded += examples.Count;
                            totalGames++;
                        }
                    }
                    else if (!auditPassed)
                    {
                        logAction?.Invoke($"[审计拒绝] {fileName}: 存在非法走法，已从训练集中排除。");
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
            var session = new GameRuleSession();
            int moveIdx = 1;

            foreach (var ucci in ucciHistory)
            {
                // 审计时允许跳过长打/长捉检测，因为外部棋谱可能遵循不同细则。
                // 但物理走法与送将判断仍必须合法。
                if (!session.TryResolveUcci(ucci, out var move, out string validationResult, skipPerpetualCheck: true))
                {
                    string pieceName = validationResult == "无效UCCI"
                        ? "未知棋子"
                        : Board.GetPieceName(session.Board.GetPiece(move.From));
                    string msg = $"第{moveIdx}步 {ucci} 疑似非法 ({pieceName} {validationResult})";

                    logAction?.Invoke($"[审计警告] {fileName} {msg}");
                    onAuditFailure?.Invoke(session.MoveHistory.ToList(), move, msg);
                    return false;
                }

                session.ApplyMove(move, ucci);
                moveIdx++;
            }

            return true;
        }

        public void AddRange(List<TrainingExample> newExamples, bool saveToDisk = true)
        {
            if (saveToDisk)
                SaveExamples(newExamples);

            lock (_buffer)
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
            lock (_buffer)
            {
                if (_count == 0)
                    return new List<TrainingExample>();

                int sampleCount = Math.Min(batchSize, _count);
                var batch = new List<TrainingExample>(sampleCount);

                var indices = new HashSet<int>();
                while (indices.Count < sampleCount)
                {
                    indices.Add(_random.Next(_count));
                }

                foreach (var i in indices)
                    batch.Add(_buffer[i]);

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
