using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using ChineseChessAI.Utils; // 引入我们新建的万能转换器
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Channels;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace ChineseChessAI
{
    public partial class MainWindow : Window
    {
        private Button[] _cellButtons = new Button[90];
        private bool _isTraining = false;

        private Channel<List<Move>> _replayChannel;

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoardUI();
            this.Loaded += (s, e) => DrawBoardLines();

            _replayChannel = Channel.CreateBounded<List<Move>>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });

            _ = Task.Run(StartReplayLoopAsync);
        }

        private void InitializeBoardUI()
        {
            ChessBoardGrid.Children.Clear();
            Style pieceStyle = (Style)this.FindResource("ChessPieceStyle");

            for (int i = 0; i < 90; i++)
            {
                var btn = new Button
                {
                    Style = pieceStyle,
                    Content = "",
                    FontSize = 26,
                    FontFamily = new FontFamily("KaiTi"),
                    FontWeight = FontWeights.Bold,
                    Margin = new Thickness(2),
                    Tag = null
                };
                _cellButtons[i] = btn;
                ChessBoardGrid.Children.Add(btn);
            }
        }

        private void DrawBoardLines()
        {
            if (ChessLinesCanvas == null)
                return;
            ChessLinesCanvas.Children.Clear();
            double w = ChessLinesCanvas.ActualWidth, h = ChessLinesCanvas.ActualHeight;
            double stepX = w / 9, stepY = h / 10;

            for (int i = 0; i < 10; i++)
                DrawLine(stepX / 2, i * stepY + stepY / 2, w - stepX / 2, i * stepY + stepY / 2);
            for (int i = 0; i < 9; i++)
            {
                if (i == 0 || i == 8)
                    DrawLine(i * stepX + stepX / 2, stepY / 2, i * stepX + stepX / 2, h - stepY / 2);
                else
                {
                    DrawLine(i * stepX + stepX / 2, stepY / 2, i * stepX + stepX / 2, 4 * stepY + stepY / 2);
                    DrawLine(i * stepX + stepX / 2, 5 * stepY + stepY / 2, i * stepX + stepX / 2, h - stepY / 2);
                }
            }
            DrawLine(3 * stepX + stepX / 2, stepY / 2, 5 * stepX + stepX / 2, 2 * stepY + stepY / 2);
            DrawLine(5 * stepX + stepX / 2, stepY / 2, 3 * stepX + stepX / 2, 2 * stepY + stepY / 2);
            DrawLine(3 * stepX + stepX / 2, 7 * stepY + stepY / 2, 5 * stepX + stepX / 2, 9 * stepY + stepY / 2);
            DrawLine(5 * stepX + stepX / 2, 7 * stepY + stepY / 2, 3 * stepX + stepX / 2, 9 * stepY + stepY / 2);
        }

        private void DrawLine(double x1, double y1, double x2, double y2)
        {
            ChessLinesCanvas.Children.Add(new Line { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Stroke = Brushes.Black, StrokeThickness = 1.2 });
        }

        private async Task StartReplayLoopAsync()
        {
            try
            {
                await foreach (var gameMoves in _replayChannel.Reader.ReadAllAsync())
                {
                    Log($"[观战] 开始播放新对局，步数: {gameMoves.Count}");
                    await ReplayGameInternalAsync(gameMoves);
                }
            }
            catch (ChannelClosedException) { }
        }

        private async Task ReplayGameInternalAsync(List<Move> historyMoves)
        {
            Board uiBoard = new Board();
            uiBoard.Reset();

            Dispatcher.Invoke(() => RefreshBoardOnly(uiBoard));

            foreach (var move in historyMoves)
            {
                if (_replayChannel.Reader.Count > 0)
                {
                    Log("[观战] 接收到最新战况，强制中断旧回放...");
                    break;
                }

                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                });

                await Task.Delay(400);

                if (_replayChannel.Reader.Count > 0)
                    break;

                uiBoard.Push(move.From, move.To);
                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                    _cellButtons[move.To].Tag = "To";
                });

                await Task.Delay(600);
            }

            if (_replayChannel.Reader.Count == 0)
                await Task.Delay(3000);
        }

        private void RefreshBoardOnly(Board board)
        {
            for (int i = 0; i < 90; i++)
            {
                sbyte p = board.GetPiece(i);
                _cellButtons[i].Content = Board.GetPieceName(p);
                _cellButtons[i].Foreground = p > 0 ? Brushes.Red : Brushes.Black;
                _cellButtons[i].Tag = null;
            }
            MoveListLog.Text = board.GetMoveHistoryString();
        }

        // ================= 单局功能事件 =================

        private async void OnStartTrainingClick(object sender, RoutedEventArgs e)
        {
            if (_isTraining)
                return;
            _isTraining = true;
            StartBtn.IsEnabled = false;

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
                        if (!_isTraining)
                            break;

                        Log($"\n--- [迭代: 第 {iter} 轮] 正在后台极速对弈... ---");

                        GameResult result = await selfPlay.RunGameAsync(null);

                        if (result.MoveHistory != null && result.MoveHistory.Count > 0)
                            _replayChannel.Writer.TryWrite(result.MoveHistory);

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
                            Dispatcher.Invoke(() => LossLabel.Text = loss.ToString("F4"));
                            ModelManager.SaveModel(model, modelPath);
                            Log($"[训练] 完成，当前 Loss: {loss:F4}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"[致命错误] {ex.Message}");
                }
                finally
                {
                    _isTraining = false;
                    Dispatcher.Invoke(() => StartBtn.IsEnabled = true);
                }
            });
        }

        private void OnReplayLastClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("极速模式下，后台的最新对局已经自动推送至棋盘频道。");
        }

        private void OnLoadFileClick(object sender, RoutedEventArgs e)
        {
            if (_isTraining)
                return;

            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "选择棋谱文件",
                Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*",
                InitialDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs")
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string fileContent = File.ReadAllText(openFileDialog.FileName);
                    string ucciRecord = "";
                    var lines = fileContent.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        if (line.StartsWith("棋谱:"))
                        {
                            ucciRecord = line.Substring(3).Trim();
                            break;
                        }
                    }

                    var moveList = new List<Move>();
                    var movesStr = ucciRecord.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var mStr in movesStr)
                    {
                        var move = NotationConverter.UcciToMove(mStr);
                        if (move != null)
                            moveList.Add(move.Value);
                    }

                    _replayChannel.Writer.TryWrite(moveList);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"解析失败: {ex.Message}");
                }
            }
        }

        // ================= 万能监督学习入口 (智能格式路由) =================

        private async void OnLoadDatasetClick(object sender, RoutedEventArgs e)
        {
            if (_isTraining)
            {
                MessageBox.Show("请先停止当前训练！", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "选择棋谱数据集",
                Filter = "支持的数据集 (*.csv;*.pgn;*.txt)|*.csv;*.pgn;*.txt|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                StartBtn.IsEnabled = false;
                string filePath = openFileDialog.FileName;
                string extension = Path.GetExtension(filePath).ToLower();

                Log($"[系统] 正在分析文件格式: {Path.GetFileName(filePath)} ...");

                await Task.Run(() =>
                {
                    try
                    {
                        // 智能路由策略
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
                            Log($"[错误] 不支持的文件扩展名: {extension}");
                        }
                    }
                    catch (Exception ex)
                    {
                        Log($"[解析致命错误] {ex.Message}");
                    }
                });

                StartBtn.IsEnabled = true;
            }
        }

        // ================= 核心推演与训练引擎 =================

        /// <summary>
        /// 核心方法：不论输入的是 WXF 还是 PGN UCCI，一律通过转换器翻译成标准 UCCI 提取
        /// </summary>
        private bool ProcessSingleMove(Board board, string rawMove, MoveGenerator generator, List<(float[] state, float[] policy, bool isRedTurn)> gameHistory)
        {
            // 1. 万能翻译机：转换为标准 UCCI
            string ucciMove = NotationConverter.ConvertToUcci(board, rawMove, generator);
            if (string.IsNullOrEmpty(ucciMove))
                return false;

            // 2. 将 UCCI 转换为内部物理引擎坐标
            Move? parsedMove = NotationConverter.UcciToMove(ucciMove);
            if (parsedMove == null)
                return false;

            // 3. 绝对物理法则校验
            var legalMoves = generator.GenerateLegalMoves(board);
            if (!legalMoves.Any(m => m.From == parsedMove.Value.From && m.To == parsedMove.Value.To))
                return false;

            // 4. 生成完美的 AI 特征张量
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
                    var allSamples = bufferToTrain.Sample(bufferToTrain.Count);
                    float epochLossSum = 0;
                    int chunksCount = 0;

                    for (int i = 0; i < allSamples.Count; i += chunkSize)
                    {
                        int currentChunkSize = Math.Min(chunkSize, allSamples.Count - i);
                        var chunk = allSamples.GetRange(i, currentChunkSize);

                        float loss = trainer.Train(chunk, epochs: 1);
                        epochLossSum += loss;
                        chunksCount++;

                        Dispatcher.Invoke(() => LossLabel.Text = (epochLossSum / chunksCount).ToString("F4"));
                    }
                    Log($"    -> Epoch {epoch}/{epochs} 完毕，当前内存块 Loss: {(epochLossSum / chunksCount):F4}");
                }

                ModelManager.SaveModel(model, modelPath);
            }
            catch (Exception ex)
            {
                Log($"[训练错误] {ex.Message}");
            }
        }

        // ================= 引擎 1：巨型 PGN / TXT 流式吞噬者 =================

        private void ProcessPgnDatasetStreaming(string filePath)
        {
            Log("[PGN 吞噬者] 正在将巨型文件读入内存...");
            string content = File.ReadAllText(filePath);

            var gameBlocks = content.Split(new[] { "[Event " }, StringSplitOptions.RemoveEmptyEntries);
            Log($"[PGN 吞噬者] 成功切割出 {gameBlocks.Length} 局大师对战！准备开启流式训练...");

            var generator = new MoveGenerator();
            int maxBufferSize = 200000; // 内存安全水位线
            var currentBuffer = new ReplayBuffer(maxBufferSize + 10000);

            int totalProcessedGames = 0, currentBatchGames = 0, trainingPhase = 1;

            foreach (var block in gameBlocks)
            {
                float resultValue = 0.0f;
                bool hasExplicitResult = false;
                var resultMatch = System.Text.RegularExpressions.Regex.Match(block, @"\[Result\s+""(.*?)""\]");
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

                string moveText = System.Text.RegularExpressions.Regex.Replace(block, @"\[.*?\]", "");
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
                        isGameValid = false;
                        break;
                    }
                }

                if (!hasExplicitResult)
                    resultValue = GetMaterialValue(board);

                if (isGameValid && gameHistory.Count > 10)
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

            if (currentBuffer.Count > 1000)
            {
                Log($"[PGN 吞噬者] 终章：清空剩余的 {currentBuffer.Count} 个样本...");
                ExecuteSupervisedTrainingChunk(currentBuffer, epochs: 2);
            }

            Log($"[PGN 吞噬者] 终极封神！总共吞噬了 {totalProcessedGames} 局高质量大师谱！您的 AI 已经无人能敌！");
        }

        // ================= 引擎 2：Kaggle CSV 乱序提取器 =================

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

            if (buffer.Count < 128)
            {
                Log($"[监督学习] 严重警告：有效样本过少 ({buffer.Count} 个)。");
                return;
            }

            Log($"[监督学习] 提取完毕！提取 {successGames} 局，生成 {buffer.Count} 个黄金样本！");
            ExecuteSupervisedTrainingChunk(buffer, epochs: 10);
        }

        // ================= 辅助方法 =================

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

        private void Log(string msg)
        {
            Dispatcher.Invoke(() =>
            {
                LogBox.AppendText($"{DateTime.Now:HH:mm:ss} - {msg}\n");
                LogBox.ScrollToEnd();
            });
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