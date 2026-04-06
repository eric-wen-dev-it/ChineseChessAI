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
        public event Action<string> OnLog;
        public event Action<float> OnLossUpdated;
        public event Action<List<Move>, int> OnReplayRequested;
        public event Action OnTrainingStopped;
        public event Action<string> OnError;

        public bool IsTraining { get; private set; } = false;
        public ReplayBuffer MasterBuffer { get; private set; } = new ReplayBuffer(500000, "data/master_data");
        public ReplayBuffer LeagueBuffer { get; private set; } = new ReplayBuffer(200000, "data/league_data");

        private LeagueManager _leagueManager;
        private static readonly object _gpuTrainingLock = new object();
        private static readonly ConcurrentDictionary<string, object> _fileLocks = new();
        
        private readonly ConcurrentDictionary<int, Lazy<PersistentAgent>> _agentPool = new();
        private readonly ConcurrentQueue<int> _activeAgentQueue = new();
        private const int MaxActiveAgents = 100;

        private readonly ConcurrentDictionary<int, SemaphoreSlim> _agentActiveLocks = new();
        private SemaphoreSlim GetAgentActiveLock(int id) => _agentActiveLocks.GetOrAdd(id, _ => new SemaphoreSlim(1, 1));
        private object GetFileLock(string path) => _fileLocks.GetOrAdd(path, _ => new object());

        public void StopTraining() => IsTraining = false;

        public async Task StartLeagueTrainingAsync(int populationSize = 50, int maxMoves = 150, int exploreMoves = 40, float materialBias = 0.1f)
        {
            if (IsTraining) return;
            IsTraining = true;

            _leagueManager = new LeagueManager(populationSize);
            foreach (var lazyAgent in _agentPool.Values) if (lazyAgent.IsValueCreated) lazyAgent.Value.Dispose();
            _agentPool.Clear();

            await Task.Run(async () =>
            {
                try
                {
                    Log($"=== 万王之王：{populationSize} 智能体联赛启动 ===");
                    Log($"[配置] 步数上限: {maxMoves} | 高温探索: {exploreMoves} | 破冰偏置: {materialBias}");

                    MasterBuffer.LoadOldSamples(int.MaxValue);
                    LeagueBuffer.LoadOldSamples(int.MaxValue);

                    const int maxParallelGames = 4;
                    int gameCounter = 0;
                    int nextLogAt = 10;
                    int nextTrainAt = 20;

                    var semaphore = new SemaphoreSlim(maxParallelGames);
                    var gameTasks = new ConcurrentBag<Task>();

                    while (IsTraining)
                    {
                        await semaphore.WaitAsync();

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
                                        
                                        bool aIsRed = Random.Shared.Next(2) == 0;
                                        var result = await selfPlay.RunGameAsync(aIsRed, null);

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

                                            Log($"[对阵] Agent_{agentMetaA.Id}(ELO:{agentMetaA.Elo:F0} DNA:S{agentMetaA.MctsSimulations}/C{agentMetaA.Cpuct:F1}/T{agentMetaA.Temperature:F1}) " +
                                                $"VS Agent_{agentMetaB.Id}(ELO:{agentMetaB.Elo:F0} DNA:S{agentMetaB.MctsSimulations}/C{agentMetaB.Cpuct:F1}/T{agentMetaB.Temperature:F1}) " +
                                                $"| {result.ResultStr} | {result.MoveCount}步");

                                            Interlocked.Increment(ref gameCounter);
                                            OnReplayRequested?.Invoke(result.MoveHistory, currentMaxMoves);
                                        }
                                    }
                                    finally { lockSecond.Release(); }
                                }
                                finally { lockFirst.Release(); }
                            }
                            catch (Exception ex) { Log($"[对局异常] {ex.Message}"); }
                            finally { semaphore.Release(); }
                        });

                        gameTasks.Add(gameTask);
                        if (gameTasks.Count > 20)
                        {
                            var completed = gameTasks.Where(t => t.IsCompleted).ToList();
                            foreach (var c in completed) gameTasks.TryTake(out _);
                        }

                        if (gameCounter >= nextTrainAt)
                        {
                            nextTrainAt += 20;
                            _ = Task.Run(() => PerformDiverseTraining());
                        }

                        if (gameCounter >= nextLogAt)
                        {
                            nextLogAt += 10;
                            _leagueManager.SaveMetadata();
                            var top = _leagueManager.GetTopAgents(5);
                            Log("--- [当前排名 Top 5] ---");
                            foreach (var t in top) Log($"ID:{t.Id} ELO:{t.Elo:F0} 胜率:{(t.Wins*100.0/Math.Max(1, t.GamesPlayed)):F1}%");
                            GC.Collect();
                        }
                    }
                }
                catch (Exception ex) { OnError?.Invoke($"[系统故障] {ex.Message}"); }
                finally { IsTraining = false; _leagueManager?.SaveMetadata(); OnTrainingStopped?.Invoke(); }
            });
        }

        private PersistentAgent GetOrAddAgent(AgentMetadata meta)
        {
            return _agentPool.GetOrAdd(meta.Id, id => new Lazy<PersistentAgent>(() =>
            {
                var pa = new PersistentAgent();
                lock (GetFileLock(meta.ModelPath)) if (File.Exists(meta.ModelPath)) pa.Model.load(meta.ModelPath);
                
                _activeAgentQueue.Enqueue(id);
                while (_activeAgentQueue.Count > MaxActiveAgents)
                {
                    if (_activeAgentQueue.TryDequeue(out int oldId) && _agentPool.TryRemove(oldId, out var oldLazy))
                    {
                        if (oldLazy.IsValueCreated) oldLazy.Value.Dispose();
                        // 【改进 P2】：同步清理信号量，防止无界增长
                        _agentActiveLocks.TryRemove(oldId, out _);
                    }
                }
                return pa;
            }, LazyThreadSafetyMode.ExecutionAndPublication)).Value;
        }

        private void PerformDiverseTraining()
        {
            lock (_gpuTrainingLock)
            {
                const int batchSize = 128;
                const float masterRatio = 0.4f;
                const float leagueRatio = 0.6f;

                foreach (var agentEntry in _agentPool)
                {
                    if (!agentEntry.Value.IsValueCreated) continue;
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
                                pa.Trainer.Train(mixedBatch, epochs: 1);
                                var meta = _leagueManager.GetAgentMeta(agentEntry.Key);
                                if (meta != null)
                                {
                                    lock (GetFileLock(meta.ModelPath)) ModelManager.SaveModel(pa.Model, meta.ModelPath);
                                }
                            }
                        }
                        finally { aLock.Release(); }
                    }
                }
            }
        }

        public static float CalculateMaterialScore(Board board, bool isRed) => isRed ? board.RedMaterial : board.BlackMaterial;
        public static float GetBoardAdvantage(Board board)
        {
            float diff = board.RedMaterial - board.BlackMaterial;
            return diff > 1.0f ? 1.0f : (diff < -1.0f ? -1.0f : 0.0f);
        }

        public async Task ProcessDatasetAsync(string filePath)
        {
            if (IsTraining) return;
            IsTraining = true;
            await Task.Run(() =>
            {
                try
                {
                    string ext = Path.GetExtension(filePath).ToLower();
                    if (ext == ".csv") ProcessCsvDataset(filePath);
                    else if (ext == ".pgn" || ext == ".txt") ProcessPgnDatasetStreaming(filePath);
                }
                catch (Exception ex) { OnError?.Invoke($"[解析错误] {ex.Message}"); }
                finally { IsTraining = false; OnTrainingStopped?.Invoke(); }
            });
        }

        private void ProcessPgnDatasetStreaming(string filePath)
        {
            Log("[PGN 吞噬者] 正在以流式方式解析文件...");
            var generator = new MoveGenerator();
            int maxBufferSize = 200000;
            var currentBuffer = new ReplayBuffer(maxBufferSize + 10000);
            int totalProcessedGames = 0, currentBatchGames = 0;

            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                StringBuilder blockBuilder = new StringBuilder();
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (!IsTraining) break;
                    if (line.StartsWith("[Event ") && blockBuilder.Length > 0)
                    {
                        ParseSinglePgnBlock(blockBuilder.ToString(), generator, currentBuffer, ref totalProcessedGames, ref currentBatchGames);
                        blockBuilder.Clear();
                    }
                    blockBuilder.AppendLine(line);
                }
                if (IsTraining && blockBuilder.Length > 0)
                    ParseSinglePgnBlock(blockBuilder.ToString(), generator, currentBuffer, ref totalProcessedGames, ref currentBatchGames);
            }
            Log($"[PGN 吞噬者] 解析完毕！总吞噬 {totalProcessedGames} 局。");
        }

        private void ProcessCsvDataset(string filePath)
        {
            Log("[CSV 解析] 开始读取文件...");
            var generator = new MoveGenerator();
            int totalGames = 0, batchGames = 0;
            string currentGameId = null;
            var redMoves = new List<(int turn, string move)>();
            var blackMoves = new List<(int turn, string move)>();
            var currentBuffer = new ReplayBuffer(200000);

            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                reader.ReadLine();
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var parts = line.Split(',');
                    if (parts.Length < 4) continue;
                    string gameId = parts[0].Trim();
                    if (!int.TryParse(parts[1].Trim(), out int turn)) continue;
                    string side = parts[2].Trim().ToLower();
                    string move = parts[3].Trim();

                    if (currentGameId != null && gameId != currentGameId)
                    {
                        ProcessCsvGame(redMoves, blackMoves, generator, currentBuffer, ref totalGames, ref batchGames);
                        redMoves.Clear(); blackMoves.Clear();
                    }
                    currentGameId = gameId;
                    if (side == "red") redMoves.Add((turn, move)); else blackMoves.Add((turn, move));
                }
            }
            Log($"[CSV 解析] 完成！总解析 {totalGames} 局。");
        }

        private void ParseSinglePgnBlock(string block, MoveGenerator generator, ReplayBuffer currentBuffer, ref int totalGames, ref int batchGames)
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
            var moveStrings = moveText.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            var board = new Board();
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();

            foreach (var rawMove in moveStrings)
            {
                string? ucci = NotationConverter.ConvertToUcci(board, rawMove, generator);
                if (string.IsNullOrEmpty(ucci)) break;
                if (ProcessSingleMoveWithUcci(board, ucci, generator, gameHistory)) standardizedMoves.Add(ucci);
                else break;
            }

            if (!hasExplicitResult) resultValue = GetBoardAdvantage(board); 
            if (gameHistory.Count > 10)
            {
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                var masterData = new MasterGameData(examples, standardizedMoves);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"pgn_game_{timestamp}.json"), JsonSerializer.Serialize(masterData));

                // 【核心修复 P0】：大师数据镜像增广 (物理上平衡红黑偏差)
                var flippedExamples = examples.Select(ex => {
                    // 1. 状态 180 度翻转 + 角色通道互换 (StateEncoder 处理的是 14 层数据)
                    float[] flippedState = new float[14 * 10 * 9];
                    for (int layer = 0; layer < 14; layer++) {
                        int targetLayer = layer < 7 ? layer + 7 : layer - 7; // 红黑通道对调
                        for (int r = 0; r < 10; r++) {
                            for (int c = 0; c < 9; c++) {
                                // 180 度旋转：(r, c) -> (9-r, 8-c)
                                flippedState[targetLayer * 90 + (9 - r) * 9 + (8 - c)] = ex.State[layer * 90 + r * 9 + c];
                            }
                        }
                    }
                    // 2. 策略坐标翻转
                    float[] densePi = new float[8100]; foreach (var p in ex.SparsePolicy) densePi[p.Index] = p.Prob;
                    float[] flippedPi = StateEncoder.FlipPolicy(densePi);
                    var sparseFlipped = flippedPi.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    
                    // 3. 结果取反
                    return new TrainingExample(flippedState, sparseFlipped, -ex.Value);
                }).ToList();
                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"pgn_mirror_{timestamp}.json"), JsonSerializer.Serialize(new MasterGameData(flippedExamples, new List<string>())));
                
                totalGames++; batchGames++;
            }
        }

        private void ProcessCsvGame(List<(int turn, string move)> redMoves, List<(int turn, string move)> blackMoves, MoveGenerator generator, ReplayBuffer currentBuffer, ref int totalGames, ref int batchGames)
        {
            redMoves.Sort((a, b) => a.turn.CompareTo(b.turn)); blackMoves.Sort((a, b) => a.turn.CompareTo(b.turn));
            var rawOrderedMoves = new List<string>();
            int maxTurn = Math.Max(redMoves.Count, blackMoves.Count);
            for (int i = 0; i < maxTurn; i++) { if (i < redMoves.Count) rawOrderedMoves.Add(redMoves[i].move); if (i < blackMoves.Count) rawOrderedMoves.Add(blackMoves[i].move); }

            var board = new Board();
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();
            foreach (var rawMove in rawOrderedMoves)
            {
                string? ucci = NotationConverter.ConvertToUcci(board, rawMove, generator);
                if (string.IsNullOrEmpty(ucci)) break;
                if (ProcessSingleMoveWithUcci(board, ucci, generator, gameHistory)) standardizedMoves.Add(ucci);
                else break;
            }
            if (gameHistory.Count > 10)
            {
                float resultValue = GetBoardAdvantage(board);
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();
                
                // 【同步 P0 修复】：对 CSV 数据也进行镜像增广
                var flippedExamples = examples.Select(ex => {
                    float[] flippedState = new float[14 * 10 * 9];
                    for (int layer = 0; layer < 14; layer++) {
                        int targetLayer = layer < 7 ? layer + 7 : layer - 7;
                        for (int r = 0; r < 10; r++) for (int c = 0; c < 9; c++) flippedState[targetLayer * 90 + (9 - r) * 9 + (8 - c)] = ex.State[layer * 90 + r * 9 + c];
                    }
                    float[] densePi = new float[8100]; foreach (var p in ex.SparsePolicy) densePi[p.Index] = p.Prob;
                    float[] flippedPi = StateEncoder.FlipPolicy(densePi);
                    return new TrainingExample(flippedState, flippedPi.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray(), -ex.Value);
                }).ToList();

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"csv_game_{timestamp}.json"), JsonSerializer.Serialize(new MasterGameData(examples, standardizedMoves)));
                File.WriteAllText(Path.Combine(MasterBuffer.DataDir, $"csv_mirror_{timestamp}.json"), JsonSerializer.Serialize(new MasterGameData(flippedExamples, new List<string>())));
                totalGames++; batchGames++;
            }
        }

        private bool ProcessSingleMoveWithUcci(Board board, string ucciMove, MoveGenerator generator, List<(float[] state, float[] policy, bool isRedTurn)> gameHistory)
        {
            Move? parsedMove = NotationConverter.UcciToMove(ucciMove);
            if (parsedMove == null) return false;
            var legalMoves = generator.GenerateLegalMoves(board);
            if (!legalMoves.Any(m => m.From == parsedMove.Value.From && m.To == parsedMove.Value.To)) return false;

            bool isRed = board.IsRedTurn;
            float[] stateData;
            using (torch.NewDisposeScope()) { var stateTensor = StateEncoder.Encode(board); stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray(); }

            float[] piData = new float[8100];
            int netIdx = parsedMove.Value.ToNetworkIndex();
            float epsilon = 0.05f; float backgroundProb = epsilon / legalMoves.Count;
            foreach (var m in legalMoves) { int idx = m.ToNetworkIndex(); if (idx >= 0 && idx < 8100) piData[idx] = backgroundProb; }
            if (netIdx >= 0 && netIdx < 8100) piData[netIdx] = (1.0f - epsilon) + backgroundProb;

            gameHistory.Add((stateData, isRed ? piData : StateEncoder.FlipPolicy(piData), isRed));
            board.Push(parsedMove.Value.From, parsedMove.Value.To);
            return true;
        }

        private void Log(string msg) => OnLog?.Invoke(msg);
    }
}
