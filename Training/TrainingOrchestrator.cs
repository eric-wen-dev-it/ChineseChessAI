using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.Training
{
    public class PersistentAgent : IDisposable
    {
        public CChessNet Model { get; }
        public Trainer Trainer { get; }

        public PersistentAgent()
        {
            Model = new CChessNet();
            if (torch.cuda.is_available()) Model.to(DeviceType.CUDA);
            Trainer = new Trainer(Model);
        }

        public void Dispose()
        {
            Trainer?.Dispose();
            Model?.Dispose();
        }
    }

    public class TrainingOrchestrator
    {
        public event Action<string>? OnLog;
        public event Action<List<Move>, int, int, string>? OnReplayRequested; // 增加结果参数
        public event Action<List<Move>, Move, string>? OnAuditFailureRequested; // 审计失败演示事件
        public event Action? OnTrainingStopped;
        public event Action<string>? OnError;

        private volatile bool _isTraining = false;
        public bool IsTraining => _isTraining;
        public ReplayBuffer MasterBuffer { get; private set; }
        public ReplayBuffer LeagueBuffer { get; private set; }

        private LeagueManager _leagueManager;
        private static readonly object _gpuTrainingLock = new object();
        private static readonly ConcurrentDictionary<string, object> _fileLocks = new();
        
        private readonly ConcurrentDictionary<int, Lazy<PersistentAgent>> _agentPool = new();

        private readonly ConcurrentDictionary<int, SemaphoreSlim> _agentActiveLocks = new();
        private SemaphoreSlim GetAgentActiveLock(int id) => _agentActiveLocks.GetOrAdd(id, _ => new SemaphoreSlim(1, 1));
        private object GetFileLock(string path) => _fileLocks.GetOrAdd(path, _ => new object());

        private CancellationTokenSource? _cts;
        private Task? _currentTrainingTask;
        private Task? _backgroundLoadTask;

        public TrainingOrchestrator()
        {
            MasterBuffer = new ReplayBuffer(5000000, "data/master_data");
            LeagueBuffer = new ReplayBuffer(1000000, "data/league_data");

            MasterBuffer.OnSaveError += msg => { Log(msg); OnError?.Invoke(msg); };
            LeagueBuffer.OnSaveError += msg => { Log(msg); OnError?.Invoke(msg); };
        }

        public void StopTraining()
        {
            _isTraining = false;
            _cts?.Cancel();
        }

        public async Task StartLeagueTrainingAsync(int populationSize = 50, int maxMoves = 150, int exploreMoves = 40, float materialBias = 0.1f)
        {
            if (populationSize > 100) throw new ArgumentException("出于内存限制与并发安全考量，联赛人口数量不能超过 100。", nameof(populationSize));
            if (populationSize < 2) throw new ArgumentException("联赛人口数量必须大于等于 2。", nameof(populationSize));

            if (IsTraining) return;
            if (_currentTrainingTask != null && !_currentTrainingTask.IsCompleted)
            {
                try { await _currentTrainingTask; } catch { }
            }
            if (_backgroundLoadTask != null && !_backgroundLoadTask.IsCompleted)
            {
                try { await _backgroundLoadTask; } catch { }
            }

            _isTraining = true;
            _cts = new CancellationTokenSource();

            _leagueManager = new LeagueManager(populationSize);
            foreach (var lazyAgent in _agentPool.Values) if (lazyAgent.IsValueCreated) lazyAgent.Value.Dispose();
            _agentPool.Clear();
            _agentActiveLocks.Clear();

            DateTime startTime = DateTime.Now;
            MasterBuffer.Clear();
            LeagueBuffer.Clear();

            _currentTrainingTask = Task.Run(async () =>
            {
                try
                {
                    Log($"=== 万王之王：{populationSize} 智能体联赛启动 ===");

                    // 将数据装载放入独立的后台任务，不阻塞联赛和对局的立即启动
                    _backgroundLoadTask = Task.Run(async () =>
                    {
                        try
                        {
                            Log("[后台任务] 正在静默装载大师数据与历史联赛数据...");
                            
                            var masterTask = MasterBuffer.LoadOldSamplesAsync(int.MaxValue, logAction: Log, onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r), cancellationToken: _cts.Token, cutoffTime: startTime);
                            var leagueTask = LeagueBuffer.LoadOldSamplesAsync(int.MaxValue, logAction: Log, onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r), cancellationToken: _cts.Token, cutoffTime: startTime);
                            
                            await Task.WhenAll(masterTask, leagueTask);
                            
                            var (masterSamples, masterGames) = await masterTask;
                            var (leagueSamples, leagueGames) = await leagueTask;
                            
                            Log($"[后台装载完成] 大师数据: {masterGames} 局 ({masterSamples} 条) | 联赛数据: {leagueGames} 局 ({leagueSamples} 条)");
                        }
                        catch (Exception ex) { Log($"[后台装载异常] {ex.Message}"); }
                    }, _cts.Token);

                    const int maxParallelGames = 4;
                    int gameCounter = 0;
                    int completedGameCounter = 0;
                    int nextLogAt = 10;
                    int nextTrainAt = 20;

                    var semaphore = new SemaphoreSlim(maxParallelGames);
                    var gameTasks = new System.Collections.Concurrent.ConcurrentQueue<Task>();

                    while (IsTraining)
                    {
                        try
                        {
                            await semaphore.WaitAsync(_cts.Token);
                        }
                        catch (OperationCanceledException)
                        {
                            break;
                        }

                        if (!IsTraining || _cts.Token.IsCancellationRequested)
                        {
                            semaphore.Release();
                            break;
                        }

                        var gameTask = Task.Run(async () =>
                        {
                            try
                            {
                                var (agentMetaA, agentMetaB) = _leagueManager.PickMatch();
                                int currentMaxMoves = (int)(maxMoves * (0.7 + Random.Shared.NextDouble() * 0.8));

                                var (firstMeta, secondMeta) = agentMetaA.Id < agentMetaB.Id ? (agentMetaA, agentMetaB) : (agentMetaB, agentMetaA);
                                var lockFirst = GetAgentActiveLock(firstMeta.Id);
                                var lockSecond = GetAgentActiveLock(secondMeta.Id);

                                await lockFirst.WaitAsync();
                                try
                                {
                                    await lockSecond.WaitAsync();
                                    try
                                    {
                                        var pAgentA = GetOrAddAgent(agentMetaA);
                                        var pAgentB = GetOrAddAgent(agentMetaB);

                                        using var engineA = new MCTSEngine(pAgentA.Model, batchSize: 16, cPuct: agentMetaA.Cpuct);
                                        using var engineB = new MCTSEngine(pAgentB.Model, batchSize: 16, cPuct: agentMetaB.Cpuct);

                                        var selfPlay = new SelfPlay(engineA, engineB, currentMaxMoves, exploreMoves, materialBias, 
                                                                    lowTempA: agentMetaA.Temperature, lowTempB: agentMetaB.Temperature, 
                                                                    simsA: agentMetaA.MctsSimulations, simsB: agentMetaB.MctsSimulations);
                                        
                                        int currentId = Interlocked.Increment(ref gameCounter);
                                        Log($"[对局 #{currentId} 开始] Agent_{agentMetaA.Id}(ELO:{agentMetaA.Elo:F0} DNA:S{agentMetaA.MctsSimulations}/C{agentMetaA.Cpuct:F1}/T{agentMetaA.Temperature:F1}) " +
                                            $"VS Agent_{agentMetaB.Id}(ELO:{agentMetaB.Elo:F0} DNA:S{agentMetaB.MctsSimulations}/C{agentMetaB.Cpuct:F1}/T{agentMetaB.Temperature:F1})");

                                        bool aIsRed = Random.Shared.Next(2) == 0;
                                        var result = await selfPlay.RunGameAsync(aIsRed, null, _cts.Token);

                                        if (result.IsSuccess)
                                        {
                                            float resA = result.ResultStr == "平局" ? 0 : (result.ResultStr == (aIsRed ? "红胜" : "黑胜") ? 1.0f : -1.0f);
                                            
                                            // 【修复 P1】：捕获赛前 ELO 以确保更新公平
                                            double eloABefore = agentMetaA.Elo;
                                            double eloBBefore = agentMetaB.Elo;

                                            lock(_leagueManager)
                                            {
                                                _leagueManager.UpdateResult(agentMetaA.Id, resA, eloBBefore);
                                                _leagueManager.UpdateResult(agentMetaB.Id, -resA, eloABefore);
                                            }

                                            if (result.ExamplesA.Count > 0) LeagueBuffer.AddRange(result.ExamplesA);
                                            if (result.ExamplesB.Count > 0) LeagueBuffer.AddRange(result.ExamplesB);

                                            Log($"[对局 #{currentId} 结束] Agent_{agentMetaA.Id}(ELO:{agentMetaA.Elo:F0}) VS Agent_{agentMetaB.Id}(ELO:{agentMetaB.Elo:F0}) | {result.ResultStr} | {result.MoveCount}步");

                                            OnReplayRequested?.Invoke(result.MoveHistory, currentMaxMoves, currentId, result.ResultStr);
                                        }
                                        else if (result.EndReason != "训练被强制终止")
                                        {
                                            throw new Exception($"对弈失败 - {result.EndReason}");
                                        }
                                    }
                                    finally 
                                    { 
                                        Interlocked.Increment(ref completedGameCounter);
                                        lockSecond.Release(); 
                                    }
                                }
                                finally { lockFirst.Release(); }
                            }
                            catch (Exception ex) { Log($"[对局异常] {ex.Message}"); }
                            finally { semaphore.Release(); }
                        });

                        gameTasks.Enqueue(gameTask);
                        while (gameTasks.Count > 20 && gameTasks.TryPeek(out var first) && first.IsCompleted)
                        {
                            gameTasks.TryDequeue(out _);
                        }

                        if (completedGameCounter >= nextTrainAt)
                        {
                            nextTrainAt += 20;
                            var trainTask = PerformDiverseTrainingAsync(_cts.Token);
                            gameTasks.Enqueue(trainTask);
                        }

                        if (completedGameCounter >= nextLogAt)
                        {
                            nextLogAt += 10;
                            _leagueManager.SaveMetadata();
                            var top = _leagueManager.GetTopAgents(5);
                            Log("--- [当前排名 Top 5] ---");
                            foreach (var t in top) Log($"ID:{t.Id} ELO:{t.Elo:F0} 胜率:{(t.Wins*100.0/Math.Max(1, t.GamesPlayed)):F1}%");
                            // 【优化 P3 #8】：移除阻塞式 GC.Collect()，交给 .NET 自动管理
                        }
                    }

                    try
                    {
                        await Task.WhenAll(gameTasks.Where(t => t != null && !t.IsCompleted).ToArray());
                    }
                    catch { }
                }
                catch (Exception ex) { OnError?.Invoke($"[系统故障] {ex.Message}"); }
                finally 
                { 
                    if (_backgroundLoadTask != null && !_backgroundLoadTask.IsCompleted)
                    {
                        try { await _backgroundLoadTask; } catch { }
                    }
                    _isTraining = false; 
                    _leagueManager?.SaveMetadata(); 
                    OnTrainingStopped?.Invoke(); 
                }
            });
        }

        private PersistentAgent GetOrAddAgent(AgentMetadata meta)
        {
            return _agentPool.GetOrAdd(meta.Id, id => new Lazy<PersistentAgent>(() =>
            {
                var pa = new PersistentAgent();
                lock (GetFileLock(meta.ModelPath)) if (File.Exists(meta.ModelPath)) pa.Model.load(meta.ModelPath);
                return pa;
            }, LazyThreadSafetyMode.ExecutionAndPublication)).Value;
        }
        private async Task PerformDiverseTrainingAsync(CancellationToken token)
        {
            await Task.Run(() =>
            {
                try
                {
                    int trainedAgents = 0;
                    int skippedBusyAgents = 0;
                    int skippedUninitializedAgents = 0;
                    int skippedNoDataAgents = 0;
                    int totalSamples = 0;
                    float totalLoss = 0f;

                    Log($"[周期训练] 开始：大师样本 {MasterBuffer.Count}，联赛样本 {LeagueBuffer.Count}，候选智能体 {_agentPool.Count}");

                    lock (_gpuTrainingLock)
                    {
                        const int batchSize = 128;
                        const float masterRatio = 0.4f;
                        const float leagueRatio = 0.6f;

                        foreach (var agentEntry in _agentPool)
                        {
                            if (token.IsCancellationRequested) return;

                            if (!agentEntry.Value.IsValueCreated)
                            {
                                skippedUninitializedAgents++;
                                continue;
                            }
                            var pa = agentEntry.Value.Value;

                            var aLock = GetAgentActiveLock(agentEntry.Key);
                            if (aLock.Wait(0))
                            {
                                try
                                {
                                    var mixedBatch = new List<TrainingExample>();
                                    if (MasterBuffer.Count > 0) mixedBatch.AddRange(MasterBuffer.Sample((int)(batchSize * masterRatio)));
                                    if (LeagueBuffer.Count > 0) mixedBatch.AddRange(LeagueBuffer.Sample((int)(batchSize * leagueRatio)));
                                    
                                    if (mixedBatch.Count > 0)
                                    {
                                        float loss = pa.Trainer.Train(mixedBatch, epochs: 1);
                                        trainedAgents++;
                                        totalSamples += mixedBatch.Count;
                                        totalLoss += loss;
                                        var meta = _leagueManager.GetAgentMeta(agentEntry.Key);
                                        if (meta != null)
                                        {
                                            lock (GetFileLock(meta.ModelPath)) ModelManager.SaveModel(pa.Model, meta.ModelPath);
                                        }
                                    }
                                    else
                                    {
                                        skippedNoDataAgents++;
                                    }
                                }
                                finally { aLock.Release(); }
                            }
                            else
                            {
                                skippedBusyAgents++;
                            }
                        }
                    }

                    if (!token.IsCancellationRequested)
                    {
                        if (trainedAgents > 0)
                        {
                            Log($"[周期训练] 完成：训练 {trainedAgents} 个智能体，使用 {totalSamples} 条样本，平均损失 {totalLoss / trainedAgents:F4}");
                        }
                        else
                        {
                            Log($"[周期训练] 跳过：没有可训练批次。忙碌 {skippedBusyAgents}，未初始化 {skippedUninitializedAgents}，空批次 {skippedNoDataAgents}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"[周期训练异常] {ex.Message}");
                    Log($"[周期训练异常-堆栈] {ex}");
                    OnError?.Invoke($"[周期训练异常] {ex.Message}");
                }
            });
        }

        public static float CalculateMaterialScore(Board board, bool isRed) => isRed ? board.RedMaterial : board.BlackMaterial;
        public static float GetBoardAdvantage(Board board)
        {
            float diff = board.RedMaterial - board.BlackMaterial;
            // 【BUG 10 优化】：降低阈值至 0.5 (约半个兵)，提高 PGN 数据标注灵敏度
            return diff > 0.5f ? 1.0f : (diff < -0.5f ? -1.0f : 0.0f);
        }

        public async Task ProcessDatasetAsync(string filePath)
        {
            if (IsTraining) return;
            if (_currentTrainingTask != null && !_currentTrainingTask.IsCompleted)
            {
                try { await _currentTrainingTask; } catch { }
            }
            _isTraining = true;
            _cts = new CancellationTokenSource();
            _currentTrainingTask = Task.Run(async () =>
            {
                try
                {
                    string ext = Path.GetExtension(filePath).ToLower();
                    if (ext == ".csv") ProcessCsvDataset(filePath, _cts.Token);
                    else if (ext == ".pgn" || ext == ".txt") ProcessPgnDatasetStreaming(filePath, _cts.Token);
                }
                catch (Exception ex) { OnError?.Invoke($"[解析错误] {ex.Message}"); }
                finally 
                { 
                    if (_backgroundLoadTask != null && !_backgroundLoadTask.IsCompleted)
                    {
                        try { await _backgroundLoadTask; } catch { }
                    }
                    _isTraining = false; 
                    OnTrainingStopped?.Invoke(); 
                }
            });
        }

        private void ProcessPgnDatasetStreaming(string filePath, CancellationToken token)
        {
            Log("[PGN 吞噬者] 正在以流式方式解析文件...");
            var rules = new ChineseChessRuleEngine();
            int totalProcessedGames = 0;

            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                StringBuilder blockBuilder = new StringBuilder();
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (token.IsCancellationRequested) break;
                    if (line.StartsWith("[Event ") && blockBuilder.Length > 0)
                    {
                        ParseSinglePgnBlock(blockBuilder.ToString(), rules, ref totalProcessedGames);
                        blockBuilder.Clear();
                    }
                    blockBuilder.AppendLine(line);
                }
                if (!token.IsCancellationRequested && blockBuilder.Length > 0)
                    ParseSinglePgnBlock(blockBuilder.ToString(), rules, ref totalProcessedGames);
            }
            Log($"[PGN 吞噬者] 解析完毕！总吞噬 {totalProcessedGames} 局。");
        }

        private void ProcessCsvDataset(string filePath, CancellationToken token)
        {
            Log("[CSV 解析] 开始读取文件...");
            var rules = new ChineseChessRuleEngine();
            int totalGames = 0;
            string currentGameId = null;
            var redMoves = new List<(int turn, string move)>();
            var blackMoves = new List<(int turn, string move)>();

            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                reader.ReadLine();
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (token.IsCancellationRequested) break;
                    var parts = line.Split(',');
                    if (parts.Length < 4) continue;
                    string gameId = parts[0].Trim();
                    if (!int.TryParse(parts[1].Trim(), out int turn)) continue;
                    string side = parts[2].Trim().ToLower();
                    string move = parts[3].Trim();

                    if (currentGameId != null && gameId != currentGameId)
                    {
                        ProcessCsvGame(redMoves, blackMoves, rules, ref totalGames);
                        redMoves.Clear(); blackMoves.Clear();
                    }
                    currentGameId = gameId;
                    if (side == "red") redMoves.Add((turn, move)); else blackMoves.Add((turn, move));
                }

                if (!token.IsCancellationRequested && currentGameId != null && (redMoves.Count > 0 || blackMoves.Count > 0))
                {
                    ProcessCsvGame(redMoves, blackMoves, rules, ref totalGames);
                    redMoves.Clear(); blackMoves.Clear();
                }
            }
            Log($"[CSV 解析] 完成！总解析 {totalGames} 局。");
        }

        private void ParseSinglePgnBlock(string block, ChineseChessRuleEngine rules, ref int totalGames)
        {
            string reconstructedBlock = block.Trim();
            if (!reconstructedBlock.StartsWith("[Event ")) reconstructedBlock = "[Event " + reconstructedBlock;

            float resultValue = 0.0f;
            bool hasExplicitResult = false;
            var resultMatch = System.Text.RegularExpressions.Regex.Match(reconstructedBlock, @"\[Result\s+""(.*?)""\]");
            if (resultMatch.Success)
            {
                string resStr = resultMatch.Groups[1].Value;
                if (resStr == "1-0") { resultValue = 1.0f; hasExplicitResult = true; }
                else if (resStr == "0-1") { resultValue = -1.0f; hasExplicitResult = true; }
                else if (resStr == "1/2-1/2") { resultValue = 0.0f; hasExplicitResult = true; }
            }

            string moveText = System.Text.RegularExpressions.Regex.Replace(reconstructedBlock, @"\[[^\]]*\]", "");
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"\{[^}]*\}", "");
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"\b\d+\.", "");
            // 去掉标准 PGN movetext 末尾的结果标记，防止被误判为非法着法导致 isComplete=false
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"(1-0|0-1|1/2-1/2|\*)\s*$", "");
            var moveStrings = moveText.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            var session = new GameRuleSession(rules);
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();
            bool isComplete = true;

            foreach (var rawMove in moveStrings)
            {
                if (ProcessSingleNotationMove(session, rawMove, gameHistory, out string normalizedUcci))
                {
                    standardizedMoves.Add(normalizedUcci);
                }
                else
                {
                    isComplete = false;
                    break;
                }
            }

            // 【数据质量修复】：如果对局截断，则不再信任整局结果，改用当前的材料差估分
            if (!isComplete)
            {
                resultValue = GetBoardAdvantage(session.Board);
            }
            else if (!hasExplicitResult) 
            {
                resultValue = GetBoardAdvantage(session.Board); 
            }

            if (isComplete && gameHistory.Count > 10)
            {
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                var masterData = new MasterGameData(examples, standardizedMoves);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string guid = Guid.NewGuid().ToString("N");
                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"pgn_game_{timestamp}_{guid}.json"), JsonSerializer.Serialize(masterData));

                // 【核心修复 P0】：大师数据镜像增广 (使用 P3 #9 优化后的稀疏翻转)
                var flippedExamples = examples.Select(ex => {
                    // 1. 策略坐标翻转
                    var sparseFlipped = StateEncoder.FlipPolicySparse(ex.SparsePolicy);
                    
                    // 2. State翻转：180度旋转 + 敌我身份交换
                    float[] flippedState = new float[1260];
                    for (int plane = 0; plane < 14; plane++)
                    {
                        int newPlane = plane < 7 ? plane + 7 : plane - 7;
                        for (int i = 0; i < 90; i++)
                        {
                            flippedState[newPlane * 90 + i] = ex.State[plane * 90 + (89 - i)];
                        }
                    }

                    // 3. Value取反：视角转换后胜负颠倒
                    return new TrainingExample(flippedState, sparseFlipped, -ex.Value);
                }).ToList();

                var flippedMoves = standardizedMoves.Select(m => {
                    if (m.Length < 4) return m;
                    char c1 = (char)('a' + 'i' - m[0]);
                    char r1 = (char)('0' + '9' - m[1]);
                    char c2 = (char)('a' + 'i' - m[2]);
                    char r2 = (char)('0' + '9' - m[3]);
                    return $"{c1}{r1}{c2}{r2}{(m.Length > 4 ? m.Substring(4) : "")}";
                }).ToList();

                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"pgn_mirror_{timestamp}_{guid}.json"), JsonSerializer.Serialize(new MasterGameData(flippedExamples, flippedMoves)));
                
                totalGames++;
            }
        }

        private void ProcessCsvGame(List<(int turn, string move)> redMoves, List<(int turn, string move)> blackMoves, ChineseChessRuleEngine rules, ref int totalGames)
        {
            redMoves.Sort((a, b) => a.turn.CompareTo(b.turn)); blackMoves.Sort((a, b) => a.turn.CompareTo(b.turn));
            var rawOrderedMoves = new List<string>();
            int maxTurn = Math.Max(redMoves.Count, blackMoves.Count);
            for (int i = 0; i < maxTurn; i++) { if (i < redMoves.Count) rawOrderedMoves.Add(redMoves[i].move); if (i < blackMoves.Count) rawOrderedMoves.Add(blackMoves[i].move); }

            var session = new GameRuleSession(rules);
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();
            bool isComplete = true;
            foreach (var rawMove in rawOrderedMoves)
            {
                if (ProcessSingleNotationMove(session, rawMove, gameHistory, out string normalizedUcci))
                {
                    standardizedMoves.Add(normalizedUcci);
                }
                else
                {
                    isComplete = false;
                    break;
                }
            }
            if (isComplete && gameHistory.Count > 10)
            {
                float resultValue = GetBoardAdvantage(session.Board);
                // 注意：对于 CSV，我们本身就没有 Header Result，所以始终使用 GetBoardAdvantage 是安全的。
                // 但为了逻辑一致性，我们在这里显式体现。
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();
                
                // 【同步 P0 修复】：对 CSV 数据也进行镜像增广 (使用稀疏翻转优化)
                var flippedExamples = examples.Select(ex => {
                    // 1. 策略坐标翻转
                    var sparseFlipped = StateEncoder.FlipPolicySparse(ex.SparsePolicy);
                    
                    // 2. State翻转：180度旋转 + 敌我身份交换
                    float[] flippedState = new float[1260];
                    for (int plane = 0; plane < 14; plane++)
                    {
                        int newPlane = plane < 7 ? plane + 7 : plane - 7;
                        for (int i = 0; i < 90; i++)
                        {
                            flippedState[newPlane * 90 + i] = ex.State[plane * 90 + (89 - i)];
                        }
                    }

                    // 3. Value取反：视角转换后胜负颠倒
                    return new TrainingExample(flippedState, sparseFlipped, -ex.Value);
                }).ToList();

                var flippedMoves = standardizedMoves.Select(m => {
                    if (m.Length < 4) return m;
                    char c1 = (char)('a' + 'i' - m[0]);
                    char r1 = (char)('0' + '9' - m[1]);
                    char c2 = (char)('a' + 'i' - m[2]);
                    char r2 = (char)('0' + '9' - m[3]);
                    return $"{c1}{r1}{c2}{r2}{(m.Length > 4 ? m.Substring(4) : "")}";
                }).ToList();

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string guid = Guid.NewGuid().ToString("N");
                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"csv_game_{timestamp}_{guid}.json"), JsonSerializer.Serialize(new MasterGameData(examples, standardizedMoves)));
                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"csv_mirror_{timestamp}_{guid}.json"), JsonSerializer.Serialize(new MasterGameData(flippedExamples, flippedMoves)));
                totalGames++;
            }
        }

        private bool ProcessSingleNotationMove(GameRuleSession session, string rawMove, List<(float[] state, float[] policy, bool isRedTurn)> gameHistory, out string normalizedUcci)
        {
            normalizedUcci = string.Empty;
            if (!session.TryResolveNotation(rawMove, out var parsedMove, out normalizedUcci, out _))
                return false;

            var board = session.Board;
            var legalMoves = session.GetLegalMoves();

            bool isRed = board.IsRedTurn;
            float[] stateData;
            using (torch.NewDisposeScope()) { var stateTensor = StateEncoder.Encode(board); stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray(); }

            float[] piData = new float[8100];
            int netIdx = parsedMove.ToNetworkIndex();
            float epsilon = 0.05f; float backgroundProb = epsilon / legalMoves.Count;
            foreach (var m in legalMoves) { int idx = m.ToNetworkIndex(); if (idx >= 0 && idx < 8100) piData[idx] = backgroundProb; }
            if (netIdx >= 0 && netIdx < 8100) piData[netIdx] = (1.0f - epsilon) + backgroundProb;

            gameHistory.Add((stateData, isRed ? piData : StateEncoder.FlipPolicy(piData), isRed));
            session.ApplyMove(parsedMove, normalizedUcci);
            return true;
        }

        private void Log(string msg) => OnLog?.Invoke(msg);
    }
}
