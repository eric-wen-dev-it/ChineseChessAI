using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
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

            // 频道容量为 1，总是丢弃旧数据，只保留最新
            _replayChannel = Channel.CreateBounded<List<Move>>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });

            // 【架构升级】：窗口启动时，就让播放器作为全局守护进程跑起来
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

        /// <summary>
        /// 全局守护进程：永远在等待频道里的数据
        /// </summary>
        private async Task StartReplayLoopAsync()
        {
            try
            {
                // 无论是否在训练，播放器永远在线等待数据
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
                // 【核心打断机制】：看一眼频道，如果有新棋局被塞进来了，立刻停止当前的旧局播放！
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

                // 再次检查是否被新数据打断
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

            // 如果没有新数据打断它，终局停留一下
            if (_replayChannel.Reader.Count == 0)
            {
                await Task.Delay(3000);
            }
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

        // ================= 控制区事件 =================

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

                        // 后台跑完一局，把棋谱丢进频道！
                        if (result.MoveHistory != null && result.MoveHistory.Count > 0)
                        {
                            _replayChannel.Writer.TryWrite(result.MoveHistory);
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
                            Log($"[对弈] 警告: 步数过短({result.MoveCount})，视为无效博弈，样本已舍弃。");
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
            MessageBox.Show("极速模式下，后台的最新对局已经自动推送至棋盘频道循环播放。");
        }

        /// <summary>
        /// 【完美契合您的思路】读取文件 -> 转换成 List<Move> -> 丢入频道！
        /// </summary>
        private void OnLoadFileClick(object sender, RoutedEventArgs e)
        {
            if (_isTraining)
            {
                MessageBox.Show("训练正在全速运行中！请先停止训练，以免后台棋谱覆盖您导入的文件。", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

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

                    // 解析文件中的 UCCI 字符串
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

                    if (string.IsNullOrEmpty(ucciRecord))
                    {
                        MessageBox.Show("未能从文件中解析出棋谱数据！", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                        return;
                    }

                    Log($"[系统] 成功读取本地文件: {Path.GetFileName(openFileDialog.FileName)}");

                    // 将 UCCI 字符串转换为 List<Move> 棋局
                    var moveList = new List<Move>();
                    var movesStr = ucciRecord.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var mStr in movesStr)
                    {
                        if (mStr.Length == 4)
                        {
                            int fC = mStr[0] - 'a';
                            int fR = 9 - (mStr[1] - '0');
                            int tC = mStr[2] - 'a';
                            int tR = 9 - (mStr[3] - '0');
                            moveList.Add(new Move(fR * 9 + fC, tR * 9 + tC));
                        }
                    }

                    // 【最关键的一步】：清空当前演示（隐含在机制里），直接把棋局丢入频道展示！
                    _replayChannel.Writer.TryWrite(moveList);
                    Log($"[回放] 棋局已送入展示频道。");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"读取或解析棋谱失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void SaveMoveListToFile(string moveList, string result, string reason)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs");
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string filePath = Path.Combine(logDir, $"game_{timestamp}.txt");

                string content = $"时间: {DateTime.Now}\n" +
                                 $"结果: {result}\n" +
                                 $"原因: {reason}\n" +
                                 $"棋谱: {moveList}\n" +
                                 new string('-', 40) + "\n";
                File.WriteAllText(filePath, content);
            }
            catch (Exception) { /* 忽略日志写入错误 */ }
        }

        private void Log(string msg)
        {
            Dispatcher.Invoke(() =>
            {
                LogBox.AppendText($"{DateTime.Now:HH:mm:ss} - {msg}\n");
                LogBox.ScrollToEnd();
            });
        }


        // ================= CSV 数据集批量监督学习模块 =================

        private async void OnLoadCsvDatasetClick(object sender, RoutedEventArgs e)
        {
            if (_isTraining)
            {
                MessageBox.Show("请先停止当前训练！", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "选择 Kaggle CSV 数据集",
                Filter = "CSV files (*.csv)|*.csv|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                StartBtn.IsEnabled = false;
                Log($"[监督学习] 正在读取 CSV 文件: {Path.GetFileName(openFileDialog.FileName)} ...");

                await Task.Run(() => ProcessCsvDataset(openFileDialog.FileName));

                StartBtn.IsEnabled = true;
            }
        }

        private void ProcessCsvDataset(string filePath)
        {
            var lines = File.ReadAllLines(filePath);

            // 【终极重构】：红黑分离式存储！完全降服 Kaggle 的奇葩排版
            var games = new Dictionary<string, (Dictionary<int, string> Red, Dictionary<int, string> Black)>();

            Log("[监督学习] 正在归类对局数据并进行红黑时空交织...");
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
                        {
                            // 初始化这局棋的红黑两个大脑
                            games[gameId] = (new Dictionary<int, string>(), new Dictionary<int, string>());
                        }

                        // 各回各家
                        if (side == "red")
                            games[gameId].Red[turn] = move;
                        else
                            games[gameId].Black[turn] = move;
                    }
                }
            }

            Log($"[监督学习] 成功归类 {games.Count} 局，开始推演神经网络特征...");

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

                // 找出这局棋打了多少回合
                int maxTurn = Math.Max(
                    redMoves.Count > 0 ? redMoves.Keys.Max() : 0,
                    blackMoves.Count > 0 ? blackMoves.Keys.Max() : 0
                );

                // 【核心魔法】：拉链式交织（红一回合，黑一回合）
                for (int turn = 1; turn <= maxTurn; turn++)
                {
                    // === 1. 红方走棋 ===
                    if (redMoves.TryGetValue(turn, out string redWxf))
                    {
                        if (!ProcessSingleMove(board, redWxf, generator, gameHistory))
                            break; // 遇到无法解析的终局符或残缺，立刻截断，保留已拿到的黄金数据
                    }
                    else
                        break; // 红方数据断层

                    // === 2. 黑方走棋 ===
                    if (blackMoves.TryGetValue(turn, out string blackWxf))
                    {
                        if (!ProcessSingleMove(board, blackWxf, generator, gameHistory))
                            break;
                    }
                    else
                        break; // 黑方数据断层（很正常，可能红方上一步已经绝杀了）
                }

                // 只要交织成功了 10 步以上，这就是优质的训练素材！
                if (gameHistory.Count > 10)
                {
                    float resultValue = GetMaterialValue(board);
                    var examples = gameHistory.Select(step =>
                        new TrainingExample(step.state, step.policy, step.isRedTurn ? resultValue : -resultValue)
                    ).ToList();

                    buffer.AddRange(examples);
                    successGames++;
                }
            }

            if (buffer.Count < 128)
            {
                Log($"[监督学习] 严重警告：有效样本过少 ({buffer.Count} 个)。");
                Dispatcher.Invoke(() => StartBtn.IsEnabled = true);
                return;
            }

            Log($"[监督学习] 提取完毕！成功提取 {successGames} 局，生成了 {buffer.Count} 个黄金样本！");
            Log($"[监督学习] 开始高强度反向传播 (数据量极大，请耐心等待)...");

            try
            {
                var model = new CChessNet();
                string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
                if (File.Exists(modelPath))
                    model.load(modelPath);

                var trainer = new Trainer(model);

                // 【防显存爆炸核心】：分块喂给模型！
                // 每次只把 4096 个样本丢给 GPU 去 stack。
                // 如果您的显存依然报警，可以把这个值调成 2048 或 1024。
                int chunkSize = 4096;

                for (int epoch = 1; epoch <= 10; epoch++)
                {
                    Log($"[监督学习] --- 开始 Epoch {epoch}/10 ---");

                    // 打乱所有 50 万个黄金样本
                    var allSamples = buffer.Sample(buffer.Count);
                    float epochLossSum = 0;
                    int chunksCount = 0;

                    // 将海量数据切块，蚂蚁搬家
                    for (int i = 0; i < allSamples.Count; i += chunkSize)
                    {
                        // 高效截取当前块 (避免使用耗时的 LINQ Skip)
                        int currentChunkSize = Math.Min(chunkSize, allSamples.Count - i);
                        var chunk = allSamples.GetRange(i, currentChunkSize);

                        // GPU 只需处理这一小块数据，显存占用极小且稳定
                        float loss = trainer.Train(chunk, epochs: 1);

                        epochLossSum += loss;
                        chunksCount++;

                        // 实时更新平均 Loss
                        Dispatcher.Invoke(() =>
                        {
                            LossLabel.Text = (epochLossSum / chunksCount).ToString("F4");
                        });

                        // 每处理 20 个 Chunk 打印一次日志，避免刷屏卡死 UI
                        if (chunksCount % 20 == 0)
                        {
                            Log($"[监督学习] 进度: 批次 {chunksCount}, 当前块 Loss: {loss:F4}");
                        }
                    }

                    Log($"[监督学习] *** Epoch {epoch} 完成，本轮平均 Loss: {(epochLossSum / chunksCount):F4} ***");
                }

                ModelManager.SaveModel(model, modelPath);
                Log($"[监督学习] 大功告成！AI 完美吸收了全部 9900 局大师数据，没有发生显存溢出！");
            }
            catch (Exception ex)
            {
                Log($"[训练错误] {ex.Message}");
            }
            finally
            {
                Dispatcher.Invoke(() => StartBtn.IsEnabled = true);
            }
        }

        // 把原本臃肿的单步推演逻辑提取出来，让主代码清清爽爽
        private bool ProcessSingleMove(Board board, string wxfMove, MoveGenerator generator, List<(float[] state, float[] policy, bool isRedTurn)> gameHistory)
        {
            Move? parsedMove = ParseWxfMove(board, wxfMove, generator);
            if (parsedMove == null)
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

        /// <summary>
        /// 智能 WXF 代数谱解码器 (利用 MoveGenerator 反推)
        /// </summary>
        private Move? ParseWxfMove(Board board, string wxf, MoveGenerator generator)
        {
            var legalMoves = generator.GenerateLegalMoves(board);
            bool isRed = board.IsRedTurn;
            var candidates = new List<Move>();

            wxf = wxf.Trim().ToUpper();
            wxf = wxf.Replace('=', '.'); // 兼容某些用 = 代替 . 的平移记法

            foreach (var move in legalMoves)
            {
                sbyte piece = board.GetPiece(move.From);
                int type = Math.Abs(piece);
                char pieceChar = type switch
                {
                    1 => 'K',
                    2 => 'A',
                    3 => 'E',
                    4 => 'H',
                    5 => 'R',
                    6 => 'C',
                    7 => 'P',
                    _ => '?'
                };

                // 兼容某些 Kaggle 谱用 B (Bishop) 代替 E (Elephant)
                if (type == 3 && !wxf.Contains('E') && wxf.Contains('B'))
                    pieceChar = 'B';

                if (!wxf.Contains(pieceChar))
                    continue;

                int fromRow = move.From / 9, fromCol = move.From % 9;
                int toRow = move.To / 9, toCol = move.To % 9;

                int startFile = isRed ? (9 - fromCol) : (fromCol + 1);
                int endFile = isRed ? (9 - toCol) : (toCol + 1);

                char direction = fromRow == toRow ? '.' : ((isRed && toRow < fromRow) || (!isRed && toRow > fromRow) ? '+' : '-');
                if (!wxf.Contains(direction))
                    continue;

                int endValue = (type == 2 || type == 3 || type == 4 || direction == '.') ? endFile : Math.Abs(toRow - fromRow);

                // 【核心修复】：动态数字解析器，完美支持前后炮 (+C.5) 的单数字情况
                var digits = wxf.Where(char.IsDigit).ToArray();
                if (digits.Length == 0)
                    continue;

                // 最后一个数字永远是 目标列 或 步数
                int expectedEndVal = digits.Last() - '0';
                if (endValue != expectedEndVal)
                    continue;

                // 如果有两个数字，第一个永远是起始列
                if (digits.Length >= 2)
                {
                    int expectedStartFile = digits.First() - '0';
                    if (startFile != expectedStartFile)
                        continue;
                }

                candidates.Add(move);
            }

            if (candidates.Count == 1)
                return candidates[0];

            // 解决双炮/双兵同列的歧义 (+C.5 前炮平五, -C.5 后炮平五)
            if (candidates.Count > 1)
            {
                bool isFront = wxf.StartsWith("+") || wxf.StartsWith("F");
                bool isBack = wxf.StartsWith("-") || wxf.StartsWith("B");

                if (isFront || isBack)
                {
                    var sorted = candidates.OrderBy(m => m.From / 9).ToList();
                    if (isRed)
                        return isFront ? sorted.First() : sorted.Last(); // 红方在下方，行号越小越靠前
                    else
                        return isFront ? sorted.Last() : sorted.First(); // 黑方在上方，行号越大越靠前
                }
                return candidates[0]; // 极端情况下返回第一个，防止彻底崩溃
            }
            return null;
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