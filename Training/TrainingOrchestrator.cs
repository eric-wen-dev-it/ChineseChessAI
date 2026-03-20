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
    public class TrainingOrchestrator
    {
        public event Action<string> OnLog;
        public event Action<float> OnLossUpdated;
        public event Action<List<Move>> OnReplayRequested;
        public event Action OnTrainingStopped;
        public event Action<string> OnError;

        public bool IsTraining { get; private set; } = false;

        // 常驻内存的大师经验池（容量50万步），存放人类精华
        public ReplayBuffer MasterBuffer { get; private set; } = new ReplayBuffer(500000);

        public void StopTraining()
        {
            IsTraining = false;
        }

        // ================= 1. 自我对弈进化循环 =================

        // 接收 UI 传来的动态参数
        public async Task StartSelfPlayAsync(int maxMoves = 250, int exploreMoves = 40, float materialBias = 0.05f)
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

                    if (MasterBuffer.Count > 0)
                    {
                        Log($"[系统] 检测到大师经验池包含 {MasterBuffer.Count} 个黄金样本！将开启【混合训练模式】。");
                    }

                    var model = new CChessNet();
                    string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                    string modelPath = Path.Combine(baseDir, "best_model.pt");

                    if (File.Exists(modelPath))
                    {
                        model.load(modelPath);
                        Log("[系统] 已加载现有模型权重。");
                    }

                    var engine = new MCTSEngine(model, batchSize: 512);

                    // 将参数传递给 SelfPlay
                    var selfPlay = new SelfPlay(engine, maxMoves, exploreMoves, materialBias);
                    var buffer = new ReplayBuffer(100000);
                    buffer.LoadOldSamples();

                    var trainer = new Trainer(model);
                    var rnd = new Random();

                    for (int iter = 1; iter <= 10000; iter++)
                    {
                        if (!IsTraining)
                            break;

                        // 【新增】：在每局开始的日志中，直接打出当前的设置参数
                        Log($"\n--- [迭代: 第 {iter} 轮] 极速对弈 (限步:{maxMoves} 探索:{exploreMoves} 偏置:{materialBias:F2}) ---");

                        GameResult result = await selfPlay.RunGameAsync(null);

                        if (result.MoveHistory != null && result.MoveHistory.Count > 0)
                        {
                            OnReplayRequested?.Invoke(result.MoveHistory);
                        }

                        // 将参数打包传入保存文件的方法中
                        string moveStr = string.Join(" ", result.MoveHistory.Select(m => m.ToString()));
                        string paramInfo = $"限步={maxMoves}, 探索={exploreMoves}, 偏置={materialBias:F3}";
                        SaveMoveListToFile(moveStr, result.ResultStr, result.EndReason, paramInfo);

                        if (result.MoveCount > 10)
                        {
                            // 平局降采样 (Draw Downsampling)
                            bool isDraw = result.ResultStr == "平局";
                            bool keepSample = true;

                            if (isDraw && rnd.NextDouble() > 0.10) // 90% 的平局录像将被无情丢弃
                            {
                                keepSample = false;
                            }

                            if (keepSample)
                            {
                                buffer.AddRange(result.Examples);
                                Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 样本已存入");
                            }
                            else
                            {
                                Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 📉 平局降采样: 废棋已丢弃");
                            }
                        }
                        else
                        {
                            Log($"[对弈] 警告: 步数过短，视为无效博弈。");
                        }

                        // 将触发训练的阈值降到 3000，因为我们要混入大师数据
                        if (buffer.Count >= 3000)
                        {
                            Log($"[训练] 开始梯度下降... 当前学习率: {trainer.GetCurrentLR():F6}");

                            int batchSize = 4096;
                            List<TrainingExample> mixedBatch = new List<TrainingExample>();
                            int masterCount = 0;

                            // 按 25% 比例抽取大师数据
                            if (MasterBuffer != null && MasterBuffer.Count > 0)
                            {
                                masterCount = Math.Min((int)(batchSize * 0.25), MasterBuffer.Count);
                                mixedBatch.AddRange(MasterBuffer.Sample(masterCount));
                            }

                            // 剩余 75% 额度由最新的自我对弈数据填补
                            int selfPlayCount = Math.Min(batchSize - masterCount, buffer.Count);
                            mixedBatch.AddRange(buffer.Sample(selfPlayCount));

                            // 充分打乱混合后的样本
                            mixedBatch = mixedBatch.OrderBy(x => rnd.Next()).ToList();

                            float loss = trainer.Train(mixedBatch, epochs: 15);
                            OnLossUpdated?.Invoke(loss);

                            ModelManager.SaveModel(model, modelPath);
                            Log($"[训练] 完成，当前 Loss: {loss:F4} (其中 {masterCount} 个大师样本，{selfPlayCount} 个自我实战样本)");
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

                foreach (var rawMove in moveStrings)
                {
                    if (string.IsNullOrEmpty(rawMove.Trim()))
                        continue;
                    if (!ProcessSingleMove(board, rawMove, generator, gameHistory))
                        break;
                }

                if (!hasExplicitResult)
                    resultValue = GetMaterialValue(board);

                if (gameHistory.Count > 10)
                {
                    var examples = gameHistory.Select(step =>
                        new TrainingExample(step.state, step.policy, step.isRedTurn ? resultValue : -resultValue)
                    ).ToList();

                    currentBuffer.AddRange(examples, saveToDisk: false);
                    MasterBuffer.AddRange(examples, saveToDisk: false); // 保存到大师经验池

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
                Log($"[PGN 吞噬者] 终极封神！总共吞噬了 {totalProcessedGames} 局高质量大师谱！常驻大脑已加载。");
            else
                Log($"[系统] 训练已中断。");
        }

        private void ProcessCsvDataset(string filePath)
        {
            // 为节省篇幅此处折叠 CSV 解析逻辑，如果有需要请保留原代码逻辑
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

        // 【修改点】：接收了超参数字符串 paramInfo
        private void SaveMoveListToFile(string moveList, string result, string reason, string paramInfo)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs");
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);
                string filePath = Path.Combine(logDir, $"game_{DateTime.Now:yyyyMMdd_HHmmss}.txt");
                // 将配置印在头部
                string content = $"时间: {DateTime.Now}\n参数: {paramInfo}\n结果: {result}\n原因: {reason}\n棋谱: {moveList}\n" + new string('-', 40) + "\n";
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