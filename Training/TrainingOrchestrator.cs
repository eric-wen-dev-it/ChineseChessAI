using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ChineseChessAI.Training
{
    /// <summary>
    /// 训练总指挥官：接管所有的后台对弈、文件解析与模型训练逻辑
    /// </summary>
    public class TrainingOrchestrator
    {
        // === 向 UI 层汇报的事件 ===
        public event Action<string> OnLog;
        public event Action<float> OnLossUpdated;
        public event Action<List<Move>> OnReplayRequested;
        public event Action OnTrainingStopped;
        public event Action<string> OnError;

        public bool IsTraining { get; private set; } = false;

        public void StopTraining()
        {
            IsTraining = false;
        }

        // ================= 1. 自我对弈进化循环 =================

        public async Task StartSelfPlayAsync()
        {
            if (IsTraining)
                return;
            IsTraining = true;

            await Task.Run(async () =>
            {
                try
                {
                    Log("=== 进化循环已启动 (极速模式) ===");
                    var model = new CChessNet();
                    string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                    string modelPath = Path.Combine(baseDir, "best_model.pt");

                    if (File.Exists(modelPath))
                    {
                        model.load(modelPath);
                        Log("[系统] 已加载现有模型权重。");
                    }

                    var engine = new MCTSEngine(model, batchSize: 512);
                    var selfPlay = new SelfPlay(engine);
                    var buffer = new ReplayBuffer(100000);
                    buffer.LoadOldSamples();

                    var trainer = new Trainer(model);

                    for (int iter = 1; iter <= 10000; iter++)
                    {
                        if (!IsTraining)
                            break;

                        Log($"\n--- [迭代: 第 {iter} 轮] 正在后台极速对弈... ---");

                        GameResult result = await selfPlay.RunGameAsync(null);

                        if (result.MoveHistory != null && result.MoveHistory.Count > 0)
                        {
                            OnReplayRequested?.Invoke(result.MoveHistory);
                        }

                        string moveStr = string.Join(" ", result.MoveHistory.Select(m => m.ToString()));
                        SaveMoveListToFile(moveStr, result.ResultStr, result.EndReason);

                        if (result.MoveCount > 10)
                        {
                            buffer.AddRange(result.Examples);
                            Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 样本已存入");
                        }
                        else
                        {
                            Log($"[对弈] 警告: 步数过短，视为无效博弈。");
                        }

                        if (buffer.Count >= 4096)
                        {
                            Log($"[训练] 开始梯度下降... 当前学习率: {trainer.GetCurrentLR():F6}");
                            float loss = trainer.Train(buffer.Sample(4096), epochs: 15);
                            OnLossUpdated?.Invoke(loss);

                            ModelManager.SaveModel(model, modelPath);
                            Log($"[训练] 完成，当前 Loss: {loss:F4}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    OnError?.Invoke($"[致命错误] {ex.Message}");
                }
                finally
                {
                    IsTraining = false;
                    OnTrainingStopped?.Invoke();
                }
            });
        }

        // ================= 2. 万能数据集智能路由 =================

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
                    {
                        Log("[系统] 格式识别为: CSV 乱序对局表。已分配至【时空交织引擎】...");
                        ProcessCsvDataset(filePath);
                    }
                    else if (extension == ".pgn" || extension == ".txt")
                    {
                        Log("[系统] 格式识别为: PGN/TXT 巨型库。已分配至【流式吞噬引擎】...");
                        ProcessPgnDatasetStreaming(filePath);
                    }
                    else
                    {
                        OnError?.Invoke($"不支持的文件扩展名: {extension}");
                    }
                }
                catch (Exception ex)
                {
                    OnError?.Invoke($"[解析致命错误] {ex.Message}");
                }
                finally
                {
                    IsTraining = false;
                    OnTrainingStopped?.Invoke();
                }
            });
        }

        // ================= 3. 数据集解析与训练实现细节 =================

        private void ProcessPgnDatasetStreaming(string filePath)
        {
            Log("[PGN 吞噬者] 正在将巨型文件读入内存...");
            string content = File.ReadAllText(filePath);

            var gameBlocks = content.Split(new[] { "[Event " }, StringSplitOptions.RemoveEmptyEntries);
            Log($"[PGN 吞噬者] 成功切割出 {gameBlocks.Length} 局大师对战！准备开启流式训练...");

            var generator = new MoveGenerator();
            int maxBufferSize = 200000;
            var currentBuffer = new ReplayBuffer(maxBufferSize + 10000);

            int totalProcessedGames = 0, currentBatchGames = 0, trainingPhase = 1;

            foreach (var block in gameBlocks)
            {
                if (!IsTraining)
                    break;

                string reconstructedBlock = "[Event " + block;
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
                board.Reset();
                var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
                bool isGameValid = true;

                foreach (var rawMove in moveStrings)
                {
                    if (string.IsNullOrEmpty(rawMove.Trim()))
                        continue;

                    if (!ProcessSingleMove(board, rawMove, generator, gameHistory))
                    {
                        break;
                    }
                }

                if (!hasExplicitResult)
                    resultValue = GetMaterialValue(board);

                if (gameHistory.Count > 10)
                {
                    var examples = gameHistory.Select(step =>
                        new TrainingExample(step.state, step.policy, step.isRedTurn ? resultValue : -resultValue)
                    ).ToList();

                    currentBuffer.AddRange(examples, saveToDisk: false);
                    totalProcessedGames++;
                    currentBatchGames++;
                }

                if (currentBuffer.Count >= maxBufferSize)
                {
                    Log($"[PGN 吞噬者] 阶段 {trainingPhase}：缓存池已满 ({currentBuffer.Count} 样本)。开始消化...");
                    ExecuteSupervisedTrainingChunk(currentBuffer, epochs: 2);
                    Log($"[PGN 吞噬者] 阶段 {trainingPhase} 消化完毕！累计吸收 {totalProcessedGames} 局。清空肠胃...");

                    currentBuffer = new ReplayBuffer(maxBufferSize + 10000);
                    currentBatchGames = 0;
                    trainingPhase++;
                    GC.Collect();
                }
            }

            if (currentBuffer.Count > 1000 && IsTraining)
            {
                Log($"[PGN 吞噬者] 终章：清空剩余的 {currentBuffer.Count} 个样本...");
                ExecuteSupervisedTrainingChunk(currentBuffer, epochs: 2);
            }

            if (IsTraining)
                Log($"[PGN 吞噬者] 终极封神！总共吞噬了 {totalProcessedGames} 局高质量大师谱！");
            else
                Log($"[系统] 训练已中断。");
        }

        private void ProcessCsvDataset(string filePath)
        {
            var lines = File.ReadAllLines(filePath);
            var games = new Dictionary<string, (Dictionary<int, string> Red, Dictionary<int, string> Black)>();

            Log("[监督学习] 正在归类对局并交织时空...");
            foreach (var line in lines)
            {
                if (line.StartsWith("gameID", StringComparison.OrdinalIgnoreCase))
                    continue;
                var parts = line.Split(',');
                if (parts.Length >= 4)
                {
                    string gameId = parts[0].Trim(' ', '"');
                    string turnStr = parts[1].Trim(' ', '"');
                    string side = parts[2].Trim(' ', '"').ToLower();
                    string move = parts[3].Trim(' ', '"');

                    if (int.TryParse(turnStr, out int turn) && !string.IsNullOrEmpty(move))
                    {
                        if (!games.ContainsKey(gameId))
                            games[gameId] = (new Dictionary<int, string>(), new Dictionary<int, string>());
                        if (side == "red")
                            games[gameId].Red[turn] = move;
                        else
                            games[gameId].Black[turn] = move;
                    }
                }
            }

            var buffer = new ReplayBuffer(500000);
            var generator = new MoveGenerator();
            int successGames = 0;

            foreach (var kvp in games)
            {
                if (!IsTraining)
                    break;

                var redMoves = kvp.Value.Red;
                var blackMoves = kvp.Value.Black;
                if (redMoves.Count == 0)
                    continue;

                var board = new Board();
                board.Reset();
                var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();

                int maxTurn = Math.Max(redMoves.Count > 0 ? redMoves.Keys.Max() : 0, blackMoves.Count > 0 ? blackMoves.Keys.Max() : 0);

                for (int turn = 1; turn <= maxTurn; turn++)
                {
                    if (redMoves.TryGetValue(turn, out string redRaw))
                    {
                        if (!ProcessSingleMove(board, redRaw, generator, gameHistory))
                            break;
                    }
                    else
                        break;

                    if (blackMoves.TryGetValue(turn, out string blackRaw))
                    {
                        if (!ProcessSingleMove(board, blackRaw, generator, gameHistory))
                            break;
                    }
                    else
                        break;
                }

                if (gameHistory.Count > 10)
                {
                    float resultValue = GetMaterialValue(board);
                    var examples = gameHistory.Select(step =>
                        new TrainingExample(step.state, step.policy, step.isRedTurn ? resultValue : -resultValue)
                    ).ToList();
                    buffer.AddRange(examples, saveToDisk: false);
                    successGames++;
                }
            }

            if (!IsTraining)
                return;

            if (buffer.Count < 128)
            {
                Log($"[监督学习] 严重警告：有效样本过少 ({buffer.Count} 个)。");
                return;
            }

            Log($"[监督学习] 提取完毕！提取 {successGames} 局，生成 {buffer.Count} 个黄金样本！");
            ExecuteSupervisedTrainingChunk(buffer, epochs: 10);
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
            var stateTensor = StateEncoder.Encode(board);
            float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

            float[] piData = new float[8100];
            int netIdx = parsedMove.Value.ToNetworkIndex();
            if (netIdx >= 0 && netIdx < 8100)
                piData[netIdx] = 1.0f;

            float[] trainingPi = isRed ? piData : FlipPolicyForDataset(piData);
            gameHistory.Add((stateData, trainingPi, isRed));

            board.Push(parsedMove.Value.From, parsedMove.Value.To);
            return true;
        }

        private void ExecuteSupervisedTrainingChunk(ReplayBuffer bufferToTrain, int epochs)
        {
            try
            {
                var model = new CChessNet();
                string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
                if (File.Exists(modelPath))
                    model.load(modelPath);

                var trainer = new Trainer(model);
                int chunkSize = 4096;

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

                        int currentChunkSize = Math.Min(chunkSize, allSamples.Count - i);
                        var chunk = allSamples.GetRange(i, currentChunkSize);

                        float loss = trainer.Train(chunk, epochs: 1);
                        epochLossSum += loss;
                        chunksCount++;

                        OnLossUpdated?.Invoke(epochLossSum / chunksCount);
                    }
                    if (IsTraining)
                        Log($"    -> Epoch {epoch}/{epochs} 完毕，当前内存块 Loss: {(epochLossSum / chunksCount):F4}");
                }

                if (IsTraining)
                    ModelManager.SaveModel(model, modelPath);
            }
            catch (Exception ex)
            {
                OnError?.Invoke($"[训练错误] {ex.Message}");
            }
        }

        // ================= 4. 内部工具方法 =================

        private void Log(string msg) => OnLog?.Invoke(msg);

        private void SaveMoveListToFile(string moveList, string result, string reason)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs");
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);
                string filePath = Path.Combine(logDir, $"game_{DateTime.Now:yyyyMMdd_HHmmss}.txt");
                string content = $"时间: {DateTime.Now}\n结果: {result}\n原因: {reason}\n棋谱: {moveList}\n" + new string('-', 40) + "\n";
                File.WriteAllText(filePath, content);
            }
            catch (Exception) { }
        }

        private float GetMaterialValue(Board board)
        {
            float red = 0, black = 0;
            for (int i = 0; i < 90; i++)
            {
                sbyte p = board.GetPiece(i);
                if (p == 0)
                    continue;
                float val = Math.Abs(p) switch
                {
                    2 => 2,
                    3 => 2,
                    4 => 4,
                    5 => 9,
                    6 => 4.5f,
                    7 => 1,
                    _ => 0
                };
                if (p > 0)
                    red += val;
                else
                    black += val;
            }
            return red > black + 1.0f ? 1.0f : (black > red + 1.0f ? -1.0f : 0.0f);
        }

        private float[] FlipPolicyForDataset(float[] originalPi)
        {
            float[] flippedPi = new float[8100];
            for (int i = 0; i < 8100; i++)
            {
                if (originalPi[i] <= 0)
                    continue;
                int from = i / 90, to = i % 90;
                int r1 = from / 9, c1 = from % 9, r2 = to / 9, c2 = to % 9;
                int idx_f = ((9 - r1) * 9 + (8 - c1)) * 90 + ((9 - r2) * 9 + (8 - c2));
                if (idx_f >= 0 && idx_f < 8100)
                    flippedPi[idx_f] = originalPi[i];
            }
            return flippedPi;
        }
    }
}