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
    }
}