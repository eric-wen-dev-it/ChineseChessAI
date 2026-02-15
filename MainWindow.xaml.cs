using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

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
                    Margin = new Thickness(2)
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

            // 绘制横线
            for (int i = 0; i < 10; i++)
                DrawLine(stepX / 2, i * stepY + stepY / 2, w - stepX / 2, i * stepY + stepY / 2);

            // 绘制纵线 (中场断开)
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

            // 绘制九宫格斜线
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

        // MainWindow.xaml.cs

        private void UpdateBoard(Board board)
        {
            Dispatcher.Invoke(() =>
            {
                for (int i = 0; i < 90; i++)
                {
                    sbyte p = board.GetPiece(i);
                    _cellButtons[i].Content = GetPieceChar(p);
                    _cellButtons[i].Foreground = p > 0 ? Brushes.Red : Brushes.Black;
                }

                // 自动更新为 "兵1进1" 格式的序列
                MoveListLog.Text = board.GetMoveHistoryString();
            });
        }

        private string GetPieceChar(sbyte p)
        {
            if (p == 0)
                return "";
            string[] namesRed = { "", "帅", "仕", "相", "马", "车", "炮", "兵" };
            string[] namesBlack = { "", "将", "士", "象", "马", "车", "炮", "卒" };
            return p > 0 ? namesRed[p] : namesBlack[-p];
        }

        // MainWindow.xaml.cs 内部逻辑修改
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
                    if (System.IO.File.Exists("best_model.pt"))
                        model.load("best_model.pt");

                    var engine = new MCTSEngine(model, batchSize: 256);
                    var selfPlay = new SelfPlay(engine);
                    var buffer = new ReplayBuffer(50000);
                    var trainer = new Trainer(model);

                    for (int iter = 1; iter <= 1000; iter++)
                    {
                        Log($"\n--- [迭代: 第 {iter} 轮] ---");

                        // 【关键修改】接收 GameResult 对象
                        GameResult result = await selfPlay.RunGameAsync(b => UpdateBoard(b));

                        buffer.AddRange(result.Examples);

                        // 【日志增强】输出结束原因、结果和步数
                        Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 收集样本: {result.Examples.Count}");

                        if (buffer.Count >= 2048)
                        {
                            Log("[训练] 开始梯度下降...");
                            float loss = trainer.Train(buffer.Sample(1024), epochs: 5);
                            Dispatcher.Invoke(() => LossLabel.Text = loss.ToString("F4"));
                            // 1. 获取程序运行的绝对基准路径
                            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

                            // 2. 拼接成绝对路径
                            string fullPath = System.IO.Path.Combine(baseDir, "best_model.pt");

                            // 3. 使用绝对路径调用您的 ModelManager
                            ModelManager.SaveModel(model, fullPath);
                            Log($"[训练] 完成，Loss: {loss:F4}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"[错误] {ex.Message}");
                }
            });
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