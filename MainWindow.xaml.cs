using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;

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
            this.Loaded += (s, e) => DrawBoardLines();
        }

        private void InitializeBoardUI()
        {
            ChessBoardGrid.Children.Clear();
            for (int i = 0; i < 90; i++)
            {
                var btn = new Button
                {
                    Style = null,
                    Background = Brushes.Transparent,
                    BorderThickness = new Thickness(0),
                    Content = "",
                    FontSize = 24,
                    FontWeight = FontWeights.Bold
                };
                _cellButtons[i] = btn;
                ChessBoardGrid.Children.Add(btn);
            }
        }

        private void DrawBoardLines()
        {
            ChessLinesCanvas.Children.Clear();
            double w = ChessLinesCanvas.ActualWidth;
            double h = ChessLinesCanvas.ActualHeight;
            double stepX = w / 9;
            double stepY = h / 10;

            // 绘制横线与纵线
            for (int i = 0; i < 10; i++)
                DrawLine(0, i * stepY + stepY / 2, w, i * stepY + stepY / 2);
            for (int i = 0; i < 9; i++)
            {
                DrawLine(i * stepX + stepX / 2, stepY / 2, i * stepX + stepX / 2, h / 2 - stepY / 2); // 上半场
                DrawLine(i * stepX + stepX / 2, h / 2 + stepY / 2, i * stepX + stepX / 2, h - stepY / 2); // 下半场
            }
            // 补全两侧边缘纵线
            DrawLine(stepX / 2, stepY / 2, stepX / 2, h - stepY / 2);
            DrawLine(w - stepX / 2, stepY / 2, w - stepX / 2, h - stepY / 2);

            // 绘制九宫格斜线
            DrawLine(3 * stepX + stepX / 2, stepY / 2, 5 * stepX + stepX / 2, 2 * stepY + stepY / 2);
            DrawLine(5 * stepX + stepX / 2, stepY / 2, 3 * stepX + stepX / 2, 2 * stepY + stepY / 2);
            DrawLine(3 * stepX + stepX / 2, 7 * stepY + stepY / 2, 5 * stepX + stepX / 2, 9 * stepY + stepY / 2);
            DrawLine(5 * stepX + stepX / 2, 7 * stepY + stepY / 2, 3 * stepX + stepX / 2, 9 * stepY + stepY / 2);
        }

        private void DrawLine(double x1, double y1, double x2, double y2)
        {
            var line = new Line { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Stroke = Brushes.Black, StrokeThickness = 1.5 };
            ChessLinesCanvas.Children.Add(line);
        }

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
                MoveListLog.Text = board.GetMoveHistoryString();
            });
        }

        private string GetPieceChar(sbyte p)
        {
            if (p == 0)
                return "";
            string[] names = { "", "帅", "仕", "相", "马", "车", "炮", "兵" };
            string[] namesBlack = { "", "将", "士", "象", "马", "车", "炮", "卒" };
            return p > 0 ? names[p] : namesBlack[-p];
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
                    if (System.IO.File.Exists("best_model.pt"))
                        model.load("best_model.pt");

                    var engine = new MCTSEngine(model);
                    var selfPlay = new SelfPlay(engine);
                    var buffer = new ReplayBuffer(50000);
                    var trainer = new Trainer(model);

                    for (int iter = 1; iter <= 1000; iter++)
                    {
                        Log($"\n--- [迭代: 第 {iter} 轮] ---");

                        // 自我对弈阶段
                        var examples = await selfPlay.RunGameAsync(b => UpdateBoard(b));
                        buffer.AddRange(examples);
                        Log($"[对弈] 结束，收集样本数: {examples.Count} (Buffer总数: {buffer.Count})");

                        // 训练阶段
                        if (buffer.Count >= 1024)
                        {
                            Log("[训练] 开始梯度下降...");
                            float loss = trainer.Train(buffer.Sample(1024), epochs: 5);
                            Dispatcher.Invoke(() => LossLabel.Text = $"平均 Loss: {loss:F4}");
                            model.save("best_model.pt");
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