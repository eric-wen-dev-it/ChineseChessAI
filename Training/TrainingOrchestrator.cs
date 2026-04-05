using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System;
using System.Collections.Concurrent; // 【新增】修复潜在编译问题
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading; // 【新增】
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.Training
{
    // 【核心修复 BUG-1】：封装持久化智能体，绑定模型与训练器
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
            // 【核心审计修复】：释放底层动量与学习率缓存
            Trainer?.Dispose();
            Model?.Dispose();
        }
    }

    public class TrainingOrchestrator
    {
        public event Action<string> OnLog;
        public event Action<float> OnLossUpdated;
        public event Action<List<Move>, int> OnReplayRequested; // 升级：增加 int 参数表示步数上限
        public event Action OnTrainingStopped;
        public event Action<string> OnError;

        public bool IsTraining { get; private set; } = false;
        public ReplayBuffer MasterBuffer { get; private set; } = new ReplayBuffer(500000, "data/master_data");
        
        // 【核心改进】：共用联赛经验池，汇聚所有智能体的智慧
        public ReplayBuffer LeagueBuffer { get; private set; } = new ReplayBuffer(200000, "data/league_data");

        private LeagueManager _leagueManager;
        private static readonly object _gpuTrainingLock = new object(); // 全局锁：保护 GPU 计算与反向传播
        private static readonly ConcurrentDictionary<string, object> _fileLocks = new(); // 每文件锁：保护硬盘 IO
        
        // 【核心修复 BUG-1 & D-1】：持久化智能体池，使用 Lazy 确保严格的一次性实例化防泄漏
        private readonly ConcurrentDictionary<int, Lazy<PersistentAgent>> _agentPool = new();
        
        // 【核心修复 BUG-1】：智能体运行时锁，防止同一个 ID 在并行任务中被重用导致 CUDA 冲突
        private readonly ConcurrentDictionary<int, SemaphoreSlim> _agentActiveLocks = new();
        private SemaphoreSlim GetAgentActiveLock(int id) => _agentActiveLocks.GetOrAdd(id, _ => new SemaphoreSlim(1, 1));

        private object GetFileLock(string path) => _fileLocks.GetOrAdd(path, _ => new object());

        public void StopTraining()
        {
            IsTraining = false;
        }

        public async Task StartLeagueTrainingAsync(int populationSize = 50, int maxMoves = 150, int exploreMoves = 40, float materialBias = 0.6f)
        {
            if (IsTraining) return;
            IsTraining = true;

            _leagueManager = new LeagueManager(populationSize);
            
            // 清理旧池
            foreach (var lazyAgent in _agentPool.Values) 
                if (lazyAgent.IsValueCreated) lazyAgent.Value.Dispose();
            _agentPool.Clear();

            await Task.Run(async () =>
            {
                try
                {
                    Log($"=== 万王之王：{populationSize} 智能体联赛启动 ===");
                    Log($"[配置] 步数上限: {maxMoves} | 高温探索: {exploreMoves} | 破冰偏置: {materialBias}");

                    int masterLoaded = MasterBuffer.LoadOldSamples(int.MaxValue);
                    Log($"[系统] 已从大师库装载 {masterLoaded} 条高质量大师对局样本。");

                    const int trainEpochs = 3;
                    const int maxParallelGames = 4;
                    int gameCounter = 0;
                    int nextLogAt = 10;

                    while (IsTraining)
                    {
                        var tasks = new List<Task>();

                        for (int i = 0; i < maxParallelGames; i++)
                        {
                            if (!IsTraining) break;

                            tasks.Add(Task.Run(async () =>
                            {
                                var (agentMetaA, agentMetaB) = _leagueManager.PickMatch();
                                int currentMaxMoves = (int)(maxMoves * (0.7 + Random.Shared.NextDouble() * 0.8));

                                // 【核心修复】：按 ID 升序排序以彻底防止双向锁死锁
                                var (firstMeta, secondMeta) = agentMetaA.Id < agentMetaB.Id 
                                    ? (agentMetaA, agentMetaB) 
                                    : (agentMetaB, agentMetaA);

                                var lockFirst = GetAgentActiveLock(firstMeta.Id);
                                var lockSecond = GetAgentActiveLock(secondMeta.Id);

                                await lockFirst.WaitAsync();
                                try
                                {
                                    await lockSecond.WaitAsync();
                                    try
                                    {
                                        // 从池中获取持久化智能体 (仅在首次创建时加载硬盘权重，彻底消除多余 IO)
                                        // 使用 Lazy 保证高并发下的单例安全性
                                        var pAgentA = _agentPool.GetOrAdd(agentMetaA.Id, _ => new Lazy<PersistentAgent>(() => {
                                            var pa = new PersistentAgent();
                                            lock (GetFileLock(agentMetaA.ModelPath))
                                                if (File.Exists(agentMetaA.ModelPath)) pa.Model.load(agentMetaA.ModelPath);
                                            return pa;
                                        }, LazyThreadSafetyMode.ExecutionAndPublication)).Value;

                                        var pAgentB = _agentPool.GetOrAdd(agentMetaB.Id, _ => new Lazy<PersistentAgent>(() => {
                                            var pa = new PersistentAgent();
                                            lock (GetFileLock(agentMetaB.ModelPath))
                                                if (File.Exists(agentMetaB.ModelPath)) pa.Model.load(agentMetaB.ModelPath);
                                            return pa;
                                        }, LazyThreadSafetyMode.ExecutionAndPublication)).Value;

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
                                            
                                            lock(_leagueManager)
                                            {
                                                _leagueManager.UpdateResult(agentMetaA.Id, resA, agentMetaB.Elo);
                                                _leagueManager.UpdateResult(agentMetaB.Id, -resA, agentMetaA.Elo);
                                            }

                                            if (result.ExamplesA.Count > 0) LeagueBuffer.AddRange(result.ExamplesA);
                                            if (result.ExamplesB.Count > 0) LeagueBuffer.AddRange(result.ExamplesB);

                                            Log($"[对阵] Agent_{agentMetaA.Id}({agentMetaA.Elo:F0}) VS Agent_{agentMetaB.Id}({agentMetaB.Elo:F0}) | {result.ResultStr} | {result.MoveCount}步");

                                            // 反向传播进入 GPU 锁
                                            lock (_gpuTrainingLock)
                                            {
                                                const int batchSize = 128; 
                                                const float masterRatio = 0.3f; 
                                                const float leagueRatio = 0.7f; 

                                                void TrainAgent(PersistentAgent pa)
                                                {
                                                    var mixedBatch = new List<TrainingExample>();
                                                    if (MasterBuffer.Count > 0) mixedBatch.AddRange(MasterBuffer.Sample((int)(batchSize * masterRatio)));
                                                    if (LeagueBuffer.Count > 0) mixedBatch.AddRange(LeagueBuffer.Sample((int)(batchSize * leagueRatio)));
                                                    if (mixedBatch.Count > 0) pa.Trainer.Train(mixedBatch, epochs: trainEpochs);
                                                }

                                                TrainAgent(pAgentA);
                                                TrainAgent(pAgentB);

                                                lock (GetFileLock(agentMetaA.ModelPath)) ModelManager.SaveModel(pAgentA.Model, agentMetaA.ModelPath);
                                                lock (GetFileLock(agentMetaB.ModelPath)) ModelManager.SaveModel(pAgentB.Model, agentMetaB.ModelPath);
                                            }

                                            OnReplayRequested?.Invoke(result.MoveHistory, currentMaxMoves);
                                            Interlocked.Increment(ref gameCounter);
                                        }
                                    }
                                    finally { lockSecond.Release(); }
                                }
                                finally { lockFirst.Release(); }
                            }));
                        }

                        await Task.WhenAll(tasks);

                        _ = Task.Run(() => {
                            GC.Collect();
                            GC.WaitForPendingFinalizers();
                            if (torch.cuda.is_available()) torch.cuda.synchronize();
                        });

                        // 【核心修复 BUG-2】：修正排行榜刷新条件
                        if (gameCounter > 0 && gameCounter % 10 == 0)
                        {
                            _leagueManager.SaveMetadata();
                            var top = _leagueManager.GetTopAgents(5);
                            Log("--- [当前排名 Top 5] ---");
                            foreach (var t in top) Log($"ID:{t.Id} ELO:{t.Elo:F0} 胜率:{(t.Wins*100.0/Math.Max(1, t.GamesPlayed)):F1}%");
                        }
                    }
                }
                catch (OperationCanceledException) { Log("[联赛] 训练已取消。"); }
                catch (Exception ex) { OnError?.Invoke($"[联赛致命错误] {ex.Message}"); }
                finally
                {
                    IsTraining = false;
                    _leagueManager?.SaveMetadata();
                    OnTrainingStopped?.Invoke();
                }
            });
        }

        // ================= 1. 子力评估器 (已优化：使用 Board 的增量分数) =================
        public static float CalculateMaterialScore(Board board, bool isRed)
        {
            return isRed ? board.RedMaterial : board.BlackMaterial;
        }

        public static float GetBoardAdvantage(Board board)
        {
            float diff = board.RedMaterial - board.BlackMaterial;
            return diff > 1.0f ? 1.0f : (diff < -1.0f ? -1.0f : 0.0f);
        }

        // ================= 3. 数据集解析 (仅解析存盘，不再进行同步训练) =================
        public async Task ProcessDatasetAsync(string filePath)
        {
            if (IsTraining)
                return;
            IsTraining = true;

            await Task.Run(() =>
            {
                try
                {
                    string extension = Path.GetExtension(filePath).ToLower();
                    if (extension == ".csv")
                        ProcessCsvDataset(filePath);
                    else if (extension == ".pgn" || extension == ".txt")
                        ProcessPgnDatasetStreaming(filePath);
                }
                catch (Exception ex) { OnError?.Invoke($"[解析致命错误] {ex.Message}"); }
                finally { IsTraining = false; OnTrainingStopped?.Invoke(); }
            });
        }

        private void ProcessPgnDatasetStreaming(string filePath)
        {
            Log("[PGN 吞噬者] 正在以流式方式解析文件，将其转化为大师 JSON 训练库...");

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

                        if (currentBuffer.Count >= maxBufferSize)
                        {
                            Log($"[PGN 吞噬者] 已解析 {totalProcessedGames} 局，继续读取...");
                            currentBuffer = new ReplayBuffer(maxBufferSize + 10000);
                            currentBatchGames = 0;
                            GC.Collect();
                        }
                    }
                    blockBuilder.AppendLine(line);
                }

                if (IsTraining && blockBuilder.Length > 0)
                {
                    ParseSinglePgnBlock(blockBuilder.ToString(), generator, currentBuffer, ref totalProcessedGames, ref currentBatchGames);
                }
            }

            if (IsTraining)
                Log($"[PGN 吞噬者] 解析完毕！总吞噬 {totalProcessedGames} 局高质量大师谱。已存入 data/master_data/。");
        }

        private void ProcessCsvDataset(string filePath)
        {
            Log("[CSV 解析] 开始读取文件，仅进行解析并保存为 JSON 训练样本...");

            var generator = new MoveGenerator();
            int maxBufferSize = 200000;
            var currentBuffer = new ReplayBuffer(maxBufferSize + 10000);

            int totalGames = 0, batchGames = 0;
            string currentGameId = null;
            var redMoves = new List<(int turn, string move)>();
            var blackMoves = new List<(int turn, string move)>();

            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                reader.ReadLine(); // 跳过 header

                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (!IsTraining) break;

                    var parts = line.Split(',');
                    if (parts.Length < 4) continue;

                    string gameId = parts[0].Trim();
                    if (!int.TryParse(parts[1].Trim(), out int turn)) continue;
                    string side = parts[2].Trim().ToLower();
                    string move = parts[3].Trim();

                    if (currentGameId != null && gameId != currentGameId)
                    {
                        ProcessCsvGame(redMoves, blackMoves, generator, currentBuffer, ref totalGames, ref batchGames);
                        redMoves.Clear();
                        blackMoves.Clear();

                        if (currentBuffer.Count >= maxBufferSize)
                        {
                            Log($"[CSV 解析] 已解析 {totalGames} 局，继续读取...");
                            currentBuffer = new ReplayBuffer(maxBufferSize + 10000);
                            batchGames = 0;
                            GC.Collect();
                        }
                    }

                    currentGameId = gameId;
                    if (side == "red") redMoves.Add((turn, move));
                    else blackMoves.Add((turn, move));
                }

                if (currentGameId != null && IsTraining)
                    ProcessCsvGame(redMoves, blackMoves, generator, currentBuffer, ref totalGames, ref batchGames);
            }

            if (IsTraining)
                Log($"[CSV 解析] 完成！总解析 {totalGames} 局。已写入 data/master_data/。");
        }

        private void ParseSinglePgnBlock(string block, MoveGenerator generator, ReplayBuffer currentBuffer, ref int totalGames, ref int batchGames)
        {
            string reconstructedBlock = block.Trim();
            if (!reconstructedBlock.StartsWith("[Event "))
                reconstructedBlock = "[Event " + reconstructedBlock;

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
            moveText = moveText.Replace("1-0", "").Replace("0-1", "").Replace("1/2-1/2", "").Replace("*", "");

            var moveStrings = moveText.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            var board = new Board();
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();

            foreach (var rawMove in moveStrings)
            {
                if (string.IsNullOrEmpty(rawMove.Trim())) continue;
                
                // 【核心改进】：在模拟过程中直接获取转换后的标准 UCCI 字符串
                string? ucci = NotationConverter.ConvertToUcci(board, rawMove, generator);
                if (string.IsNullOrEmpty(ucci)) break;

                if (ProcessSingleMoveWithUcci(board, ucci, generator, gameHistory))
                {
                    standardizedMoves.Add(ucci);
                }
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

                // 统一存储为 MasterGameData 格式，且 MoveHistoryUcci 全是标准坐标
                var masterData = new MasterGameData(examples, standardizedMoves);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                string filePath = Path.Combine(MasterBuffer.DataDir, $"pgn_game_{timestamp}.json");
                File.WriteAllText(filePath, JsonSerializer.Serialize(masterData));

                totalGames++;
                batchGames++;
            }
        }

        private void ProcessCsvGame(List<(int turn, string move)> redMoves, List<(int turn, string move)> blackMoves,
            MoveGenerator generator, ReplayBuffer currentBuffer, ref int totalGames, ref int batchGames)
        {
            redMoves.Sort((a, b) => a.turn.CompareTo(b.turn));
            blackMoves.Sort((a, b) => a.turn.CompareTo(b.turn));

            var rawOrderedMoves = new List<string>();
            int maxTurn = Math.Max(redMoves.Count, blackMoves.Count);
            for (int i = 0; i < maxTurn; i++)
            {
                if (i < redMoves.Count) rawOrderedMoves.Add(redMoves[i].move);
                if (i < blackMoves.Count) rawOrderedMoves.Add(blackMoves[i].move);
            }

            var board = new Board();
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();

            foreach (var rawMove in rawOrderedMoves)
            {
                string? ucci = NotationConverter.ConvertToUcci(board, rawMove, generator);
                if (string.IsNullOrEmpty(ucci)) break;

                if (ProcessSingleMoveWithUcci(board, ucci, generator, gameHistory))
                {
                    standardizedMoves.Add(ucci);
                }
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

                var masterData = new MasterGameData(examples, standardizedMoves);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                string filePath = Path.Combine(MasterBuffer.DataDir, $"csv_game_{timestamp}.json");
                File.WriteAllText(filePath, JsonSerializer.Serialize(masterData));

                totalGames++;
                batchGames++;
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
            using (torch.NewDisposeScope())
            {
                var stateTensor = StateEncoder.Encode(board);
                stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();
            }

            float[] piData = new float[8100];
            int netIdx = parsedMove.Value.ToNetworkIndex();
            float epsilon = 0.05f;
            float backgroundProb = epsilon / legalMoves.Count;

            foreach (var m in legalMoves)
            {
                int idx = m.ToNetworkIndex();
                if (idx >= 0 && idx < 8100) piData[idx] = backgroundProb;
            }

            if (netIdx >= 0 && netIdx < 8100) piData[netIdx] = (1.0f - epsilon) + backgroundProb;

            float[] trainingPi = isRed ? piData : StateEncoder.FlipPolicy(piData);
            gameHistory.Add((stateData, trainingPi, isRed));
            board.Push(parsedMove.Value.From, parsedMove.Value.To);
            return true;
        }

        private void Log(string msg) => OnLog?.Invoke(msg);

        private void WriteErrorLog(string message, Exception ex)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "error_logs");
                if (!Directory.Exists(logDir)) Directory.CreateDirectory(logDir);
                string filePath = Path.Combine(logDir, $"error_{DateTime.Now:yyyyMMdd_HHmmss_fff}.txt");
                string content = $"{message}\nStackTrace:\n{ex.StackTrace}\n";
                if (ex.InnerException != null) content += $"InnerException: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}\n{ex.InnerException.StackTrace}\n";
                File.WriteAllText(filePath, content);
            }
            catch (Exception) { }
        }

        private void SaveMoveListToFile(string moveList, string result, string reason, string paramInfo)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs");
                if (!Directory.Exists(logDir)) Directory.CreateDirectory(logDir);
                string filePath = Path.Combine(logDir, $"game_{DateTime.Now:yyyyMMdd_HHmmss_fff}.txt");
                string content = $"时间: {DateTime.Now}\n参数: {paramInfo}\n结果: {result}\n原因: {reason}\n棋谱: {moveList}\n----------------------------------------\n";
                File.WriteAllText(filePath, content);
            }
            catch (Exception) { }
        }
    }
}