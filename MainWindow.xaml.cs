using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using System.IO;
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

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoardUI();
            // 订阅加载事件以绘制棋盘线条
            this.Loaded += (s, e) => DrawBoardLines();
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
            double w = ChessLinesCanvas.ActualWidth;
            double h = ChessLinesCanvas.ActualHeight;
            double stepX = w / 9;
            double stepY = h / 10;

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
            var line = new Line { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Stroke = Brushes.Black, StrokeThickness = 1.2 };
            ChessLinesCanvas.Children.Add(line);
        }

        private async Task UpdateBoardWithAnimation(Board board)
        {
            var move = board.LastMove;
            if (move == null)
            {
                Dispatcher.Invoke(() => RefreshBoardOnly(board));
                return;
            }

            Dispatcher.Invoke(() =>
            {
                RefreshBoardOnly(board);
                sbyte movingPiece = board.GetPiece(move.Value.To);
                _cellButtons[move.Value.To].Content = "";
                _cellButtons[move.Value.From].Content = Board.GetPieceName(movingPiece);
                _cellButtons[move.Value.From].Foreground = movingPiece > 0 ? Brushes.Red : Brushes.Black;
                _cellButtons[move.Value.From].Tag = "From";
                _cellButtons[move.Value.To].Tag = null;
            });

            await Task.Delay(600); // 每步走子的动画间隔

            Dispatcher.Invoke(() =>
            {
                RefreshBoardOnly(board);
                _cellButtons[move.Value.From].Tag = "From";
                _cellButtons[move.Value.To].Tag = "To";
            });
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
                    Log("=== 进化循环已启动 ===");
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
                    Log("[系统] 正在扫描磁盘旧样本...");
                    buffer.LoadOldSamples();

                    var trainer = new Trainer(model);

                    for (int iter = 1; iter <= 10000; iter++)
                    {
                        Log($"\n--- [迭代: 第 {iter} 轮] ---");

                        GameResult result = await selfPlay.RunGameAsync(async b => await UpdateBoardWithAnimation(b));

                        // 【新增】终局画面停留 1000ms 供观察结果
                        await Task.Delay(10000);

                        // 【修复】在 UI 线程安全获取文本，防止 InvalidOperationException
                        string currentMoveList = "";
                        Dispatcher.Invoke(() => {
                            currentMoveList = MoveListLog.Text;
                        });

                        // 保存棋谱到本地文件
                        SaveMoveListToFile(currentMoveList, result.ResultStr, result.EndReason);

                        if (result.MoveCount > 10)
                        {
                            buffer.AddRange(result.Examples);
                            Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 样本已存入缓存");
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
                    _isTraining = false;
                    Dispatcher.Invoke(() => StartBtn.IsEnabled = true);
                }
            });
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
            catch (Exception ex)
            {
                Console.WriteLine($"[系统] 保存棋谱文件失败: {ex.Message}");
            }
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