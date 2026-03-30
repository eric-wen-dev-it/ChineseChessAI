using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.Training
{
    public class TrainingOrchestrator
    {
        public event Action<string> OnLog;
        public event Action<float> OnLossUpdated;
        public event Action<List<Move>> OnReplayRequested;
        public event Action OnTrainingStopped;
        public event Action<string> OnError;

        public bool IsTraining { get; private set; } = false;
        public ReplayBuffer MasterBuffer { get; private set; } = new ReplayBuffer(500000,
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "master_data"));

        public void StopTraining()
        {
            IsTraining = false;
        }

        // ================= 1. 子力评估器 (统一收口，消除重复代码) =================
        public static float CalculateMaterialScore(Board board, bool isRed)
        {
            float score = 0;
            for (int i = 0; i < 90; i++)
            {
                sbyte p = board.GetPiece(i);
                if (p == 0)
                    continue;
                if ((isRed && p > 0) || (!isRed && p < 0))
                {
                    int type = Math.Abs(p);
                    score += type switch
                    {
                        1 => 0,
                        2 => 2,
                        3 => 2,
                        4 => 4,
                        5 => 9,
                        6 => 4.5f,
                        7 => 1,
                        _ => 0
                    };
                }
            }
            return score;
        }

        public static float GetBoardAdvantage(Board board)
        {
            float red = CalculateMaterialScore(board, true);
            float black = CalculateMaterialScore(board, false);
            return red > black + 1.0f ? 1.0f : (black > red + 1.0f ? -1.0f : 0.0f);
        }

        // ================= 2. 自我对弈进化循环 =================
        public async Task StartSelfPlayAsync(int maxMoves = 100, int exploreMoves = 40, float materialBias = 0.4f)
        {
            if (IsTraining)
                return;
            IsTraining = true;

            await Task.Run(async () =>
            {
                try
                {
                    Log("=== 进化循环已启动 (极速模式) ===");
                    Log($"[全局配置] 步数上限: {maxMoves} | 高温探索: 前 {exploreMoves} 步 | 破冰偏置: {materialBias:F3}");

                    var model = new CChessNet();
                    string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                    string modelPath = Path.Combine(baseDir, "best_model.pt");

                    if (File.Exists(modelPath))
                    {
                        model.load(modelPath);
                        Log("[系统] 已加载现有模型权重。");
                    }

                    var trainer = new Trainer(model);
                    using var engine = new MCTSEngine(model, batchSize: 512);
                    var selfPlay = new SelfPlay(engine, maxMoves, exploreMoves, materialBias);
                    var buffer = new ReplayBuffer(100000);
                    buffer.LoadOldSamples();
                    MasterBuffer.LoadOldSamples();
                    Log($"[系统] MasterBuffer 已加载 {MasterBuffer.Count} 条大师样本。");

                    var rnd = new Random();

                    for (int iter = 1; iter <= 10000; iter++)
                    {
                        if (!IsTraining)
                            break;

                        Log($"\n--- [迭代: 第 {iter} 轮] 极速对弈 ---");

                        GameResult result = await selfPlay.RunGameAsync(null);

                        if (result.MoveHistory != null && result.MoveHistory.Count > 0)
                            OnReplayRequested?.Invoke(result.MoveHistory);

                        string moveStr = string.Join(" ", result.MoveHistory.Select(m => m.ToString()));
                        string paramInfo = $"限步={maxMoves}, 探索={exploreMoves}, 偏置={materialBias:F3}";
                        SaveMoveListToFile(moveStr, result.ResultStr, result.EndReason, paramInfo);

                        if (result.MoveCount > 10)
                        {
                            bool isDraw = result.ResultStr == "平局";
                            bool keepSample = !(isDraw && rnd.NextDouble() > 0.65);

                            if (keepSample)
                            {
                                buffer.AddRange(result.Examples);
                                Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 样本已存入");
                            }
                            else
                            {
                                Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 📉 平局降采样丢弃");
                            }
                        }

                        if (buffer.Count >= 500)
                        {
                            Log($"[训练] 开始梯度下降... 当前学习率: {trainer.GetCurrentLR():F6}");
                            int batchSize = 512;
                            List<TrainingExample> mixedBatch = new List<TrainingExample>();
                            int masterCount = 0;

                            if (MasterBuffer != null && MasterBuffer.Count > 0)
                            {
                                masterCount = Math.Min((int)(batchSize * 0.50), MasterBuffer.Count);
                                mixedBatch.AddRange(MasterBuffer.Sample(masterCount));
                            }

                            int selfPlayCount = Math.Min(batchSize - masterCount, buffer.Count);
                            mixedBatch.AddRange(buffer.Sample(selfPlayCount));
                            mixedBatch = mixedBatch.OrderBy(x => rnd.Next()).ToList();

                            float loss = 0;
                            try
                            {
                                loss = trainer.Train(mixedBatch, epochs: 5);
                                OnLossUpdated?.Invoke(loss);
                                ModelManager.SaveModel(model, modelPath);
                                Log($"[训练] 完成，当前 Loss: {loss:F4}");
                            }
                            catch (Exception trainEx)
                            {
                                string errMsg = $"[训练错误] {DateTime.Now:yyyy-MM-dd HH:mm:ss} | Iter={iter} | Batch={mixedBatch.Count} | {trainEx.GetType().Name}: {trainEx.Message}";
                                Log(errMsg);
                                WriteErrorLog(errMsg, trainEx);
                                try
                                {
                                    if (torch.cuda.is_available())
                                        torch.cuda.synchronize();
                                }
                                catch { }
                                GC.Collect();
                            }
                        }
                    }
                }
                catch (Exception ex) { OnError?.Invoke($"[致命错误] {ex.Message}"); }
                finally { IsTraining = false; OnTrainingStopped?.Invoke(); }
            });
        }

        // ================= 3. 数据集解析 =================
        public async Task ProcessDatasetAsync(string filePath, bool trainWhileParsing = true)
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
                        ProcessCsvDataset(filePath, trainWhileParsing);
                    else if (extension == ".pgn" || extension == ".txt")
                        ProcessPgnDatasetStreaming(filePath, trainWhileParsing);
                }
                catch (Exception ex) { OnError?.Invoke($"[解析致命错误] {ex.Message}"); }
                finally { IsTraining = false; OnTrainingStopped?.Invoke(); }
            });
        }

        private void ProcessPgnDatasetStreaming(string filePath, bool trainWhileParsing = true)
        {
            Log(trainWhileParsing
                ? "[PGN 吞噬者] 正在以【真·流式】读取巨型文件，内存已免疫 OOM..."
                : "[PGN 吞噬者] 纯解析模式：只存盘，不训练...");

            var generator = new MoveGenerator();
            int maxBufferSize = 200000;
            var currentBuffer = new ReplayBuffer(maxBufferSize + 10000);

            var model = trainWhileParsing ? new CChessNet() : null;
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
            if (trainWhileParsing && File.Exists(modelPath))
                model.load(modelPath);
            var trainer = trainWhileParsing ? new Trainer(model) : null;

            int totalProcessedGames = 0, currentBatchGames = 0, trainingPhase = 1;

            // 【核心修复】：使用 StreamReader 一行行读，永不爆内存
            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                StringBuilder blockBuilder = new StringBuilder();
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!IsTraining)
                        break;

                    if (line.StartsWith("[Event ") && blockBuilder.Length > 0)
                    {
                        ParseSinglePgnBlock(blockBuilder.ToString(), generator, currentBuffer, ref totalProcessedGames, ref currentBatchGames);
                        blockBuilder.Clear();

                        if (currentBuffer.Count >= maxBufferSize)
                        {
                            if (trainWhileParsing)
                            {
                                Log($"[PGN 吞噬者] 阶段 {trainingPhase}：缓存池满 ({currentBuffer.Count})。开始消化...");
                                ExecuteSupervisedTrainingChunk(currentBuffer, epochs: 2, trainer, model, modelPath);
                                Log($"[PGN 吞噬者] 阶段 {trainingPhase} 消化完毕！累计吸收 {totalProcessedGames} 局。");
                            }
                            else
                            {
                                Log($"[PGN 吞噬者] 已解析 {totalProcessedGames} 局，继续读取...");
                            }

                            currentBuffer = new ReplayBuffer(maxBufferSize + 10000);
                            currentBatchGames = 0;
                            trainingPhase++;
                            GC.Collect();
                        }
                    }
                    blockBuilder.AppendLine(line);
                }

                // 处理最后一块
                if (IsTraining && blockBuilder.Length > 0)
                {
                    ParseSinglePgnBlock(blockBuilder.ToString(), generator, currentBuffer, ref totalProcessedGames, ref currentBatchGames);
                }
            }

            if (currentBuffer.Count > 1000 && IsTraining && trainWhileParsing)
            {
                Log($"[PGN 吞噬者] 终章：清空剩余 {currentBuffer.Count} 个样本...");
                ExecuteSupervisedTrainingChunk(currentBuffer, epochs: 2, trainer, model, modelPath);
            }

            if (IsTraining)
                Log($"[PGN 吞噬者] 终极封神！总吞噬 {totalProcessedGames} 局高质量大师谱。已写入 data/master_data/。");
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
                if (resStr == "1-0")
                {
                    resultValue = 1.0f;
                    hasExplicitResult = true;
                }
                else if (resStr == "0-1")
                {
                    resultValue = -1.0f;
                    hasExplicitResult = true;
                }
                else if (resStr == "1/2-1/2")
                {
                    resultValue = 0.0f;
                    hasExplicitResult = true;
                }
            }

            string moveText = System.Text.RegularExpressions.Regex.Replace(reconstructedBlock, @"\[[^\]]*\]", "");
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"\{[^}]*\}", "");
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"\b\d+\.", "");
            moveText = moveText.Replace("1-0", "").Replace("0-1", "").Replace("1/2-1/2", "").Replace("*", "");

            var moveStrings = moveText.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            var board = new Board();
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();

            foreach (var rawMove in moveStrings)
            {
                if (string.IsNullOrEmpty(rawMove.Trim()))
                    continue;
                if (!ProcessSingleMove(board, rawMove, generator, gameHistory))
                    break; // 遇到无法解析的残步，直接截断
            }

            if (!hasExplicitResult)
                resultValue = GetBoardAdvantage(board); // 使用统一评估器

            if (gameHistory.Count > 10)
            {
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                currentBuffer.AddRange(examples, saveToDisk: false);
                MasterBuffer.AddRange(examples, saveToDisk: true);
                totalGames++;
                batchGames++;
            }
        }

        private void ProcessCsvDataset(string filePath, bool trainWhileParsing = true)
        {
            Log(trainWhileParsing
                ? "[CSV 解析] 开始读取，解析 + 训练模式..."
                : "[CSV 解析] 开始读取，纯解析存盘模式...");

            var generator = new MoveGenerator();
            int maxBufferSize = 200000;
            var currentBuffer = new ReplayBuffer(maxBufferSize + 10000);

            CChessNet model = null;
            Trainer trainer = null;
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
            if (trainWhileParsing)
            {
                model = new CChessNet();
                if (File.Exists(modelPath)) model.load(modelPath);
                trainer = new Trainer(model);
            }

            int totalGames = 0, batchGames = 0, trainingPhase = 1;
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
                            if (trainWhileParsing)
                            {
                                Log($"[CSV 解析] 阶段 {trainingPhase}：缓存池满，开始消化...");
                                ExecuteSupervisedTrainingChunk(currentBuffer, epochs: 2, trainer, model, modelPath);
                                Log($"[CSV 解析] 阶段 {trainingPhase} 消化完毕！累计 {totalGames} 局。");
                            }
                            else
                            {
                                Log($"[CSV 解析] 已解析 {totalGames} 局，继续读取...");
                            }
                            currentBuffer = new ReplayBuffer(maxBufferSize + 10000);
                            batchGames = 0;
                            trainingPhase++;
                            GC.Collect();
                        }
                    }

                    currentGameId = gameId;
                    if (side == "red")
                        redMoves.Add((turn, move));
                    else
                        blackMoves.Add((turn, move));
                }

                // 处理最后一局
                if (currentGameId != null && IsTraining)
                    ProcessCsvGame(redMoves, blackMoves, generator, currentBuffer, ref totalGames, ref batchGames);
            }

            if (currentBuffer.Count > 1000 && IsTraining && trainWhileParsing)
            {
                Log($"[CSV 解析] 终章：清空剩余 {currentBuffer.Count} 个样本...");
                ExecuteSupervisedTrainingChunk(currentBuffer, epochs: 2, trainer, model, modelPath);
            }

            if (IsTraining)
                Log($"[CSV 解析] 完成！总解析 {totalGames} 局。已写入 data/master_data/。");
        }

        private void ProcessCsvGame(List<(int turn, string move)> redMoves, List<(int turn, string move)> blackMoves,
            MoveGenerator generator, ReplayBuffer currentBuffer, ref int totalGames, ref int batchGames)
        {
            redMoves.Sort((a, b) => a.turn.CompareTo(b.turn));
            blackMoves.Sort((a, b) => a.turn.CompareTo(b.turn));

            // 交叉排列：红1、黑1、红2、黑2…
            var orderedMoves = new List<string>();
            int maxTurn = Math.Max(redMoves.Count, blackMoves.Count);
            for (int i = 0; i < maxTurn; i++)
            {
                if (i < redMoves.Count) orderedMoves.Add(redMoves[i].move);
                if (i < blackMoves.Count) orderedMoves.Add(blackMoves[i].move);
            }

            var board = new Board();
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();

            foreach (var moveStr in orderedMoves)
            {
                if (!ProcessSingleMove(board, moveStr, generator, gameHistory))
                    break;
            }

            if (gameHistory.Count > 10)
            {
                float resultValue = GetBoardAdvantage(board);
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                currentBuffer.AddRange(examples, saveToDisk: false);
                MasterBuffer.AddRange(examples, saveToDisk: true);
                totalGames++;
                batchGames++;
            }
        }

        private bool ProcessSingleMove(Board board, string rawMove, MoveGenerator generator, List<(float[] state, float[] policy, bool isRedTurn)> gameHistory)
        {
            string ucciMove = NotationConverter.ConvertToUcci(board, rawMove, generator);
            if (string.IsNullOrEmpty(ucciMove))
                return false;

            Move? parsedMove = NotationConverter.UcciToMove(ucciMove);
            if (parsedMove == null)
                return false;

            var legalMoves = generator.GenerateLegalMoves(board);
            if (!legalMoves.Any(m => m.From == parsedMove.Value.From && m.To == parsedMove.Value.To))
                return false;

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
                if (idx >= 0 && idx < 8100)
                    piData[idx] = backgroundProb;
            }

            if (netIdx >= 0 && netIdx < 8100)
                piData[netIdx] = (1.0f - epsilon) + backgroundProb;

            float[] trainingPi = isRed ? piData : StateEncoder.FlipPolicy(piData);
            gameHistory.Add((stateData, trainingPi, isRed));
            board.Push(parsedMove.Value.From, parsedMove.Value.To);
            return true;
        }

        private void ExecuteSupervisedTrainingChunk(ReplayBuffer bufferToTrain, int epochs, Trainer trainer, CChessNet model, string modelPath)
        {
            try
            {
                int chunkSize = 1024;
                for (int epoch = 1; epoch <= epochs; epoch++)
                {
                    if (!IsTraining)
                        break;
                    var allSamples = bufferToTrain.Sample(bufferToTrain.Count);
                    float epochLossSum = 0;
                    int chunksCount = 0;

                    for (int i = 0; i < allSamples.Count; i += chunkSize)
                    {
                        if (!IsTraining)
                            break;
                        var chunk = allSamples.GetRange(i, Math.Min(chunkSize, allSamples.Count - i));
                        float loss = trainer.Train(chunk, epochs: 1);
                        epochLossSum += loss;
                        chunksCount++;
                        OnLossUpdated?.Invoke(epochLossSum / chunksCount);
                    }
                    if (IsTraining)
                        Log($"    -> Epoch {epoch}/{epochs} 完毕，块平均 Loss: {(epochLossSum / chunksCount):F4}");
                }
                if (IsTraining)
                    ModelManager.SaveModel(model, modelPath);
            }
            catch (Exception ex) { OnError?.Invoke($"[训练错误] {ex.Message}"); }
        }

        private void Log(string msg) => OnLog?.Invoke(msg);

        private void WriteErrorLog(string message, Exception ex)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "error_logs");
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);
                string filePath = Path.Combine(logDir, $"error_{DateTime.Now:yyyyMMdd_HHmmss_fff}.txt");
                string content = $"{message}\nStackTrace:\n{ex.StackTrace}\n";
                if (ex.InnerException != null)
                    content += $"InnerException: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}\n{ex.InnerException.StackTrace}\n";
                File.WriteAllText(filePath, content);
            }
            catch (Exception) { }
        }

        private void SaveMoveListToFile(string moveList, string result, string reason, string paramInfo)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs");
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);
                // 【核心修复】：增加 _fff 防止并发写文件时覆盖
                string filePath = Path.Combine(logDir, $"game_{DateTime.Now:yyyyMMdd_HHmmss_fff}.txt");
                string content = $"时间: {DateTime.Now}\n参数: {paramInfo}\n结果: {result}\n原因: {reason}\n棋谱: {moveList}\n----------------------------------------\n";
                File.WriteAllText(filePath, content);
            }
            catch (Exception) { }
        }

    }
}