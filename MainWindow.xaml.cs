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

        /// <summary>
        /// 【核心修正】分步动画逻辑：
        /// 确保两次 UI 更新之间释放控制权，让 WPF 能够渲染第一阶段。
        /// </summary>
        private async Task UpdateBoardWithAnimation(Board board)
        {
            var move = board.LastMove;
            if (move == null)
            {
                Dispatcher.Invoke(() => RefreshBoardOnly(board));
                return;
            }

            // === 阶段 1：在 UI 线程绘制“起手”状态并渲染 ===
            Dispatcher.Invoke(() =>
            {
                // 先整体刷一遍物理位置
                RefreshBoardOnly(board);

                // 核心视觉欺骗：
                // 1. 获取刚才动的那颗子（当前已在终点）
                sbyte movingPiece = board.GetPiece(move.Value.To);

                // 2. 将终点强行设为空，将起点强行恢复为该棋子
                _cellButtons[move.Value.To].Content = "";
                _cellButtons[move.Value.From].Content = Board.GetPieceName(movingPiece);
                _cellButtons[move.Value.From].Foreground = movingPiece > 0 ? Brushes.Red : Brushes.Black;

                // 3. 仅亮起起点红框
                _cellButtons[move.Value.From].Tag = "From";
                _cellButtons[move.Value.To].Tag = null;
            });

            // === 阶段 2：在后台线程等待，给 UI 线程渲染红框的时间 ===
            // 此时 UI 线程已空闲，可以完成上一轮设置的“红框+原位棋子”的绘制
            await Task.Delay(200);

            // === 阶段 3：在 UI 线程执行“落子”并亮起绿框 ===
            Dispatcher.Invoke(() =>
            {
                // 再次物理刷新，此时棋子会跳到真实的终点
                RefreshBoardOnly(board);

                // 同时补上起止点的方框标记
                _cellButtons[move.Value.From].Tag = "From";
                _cellButtons[move.Value.To].Tag = "To";
            });
        }

        /// <summary>
        /// 基础物理刷新，不处理动画延迟，不包含 Tag 逻辑
        /// </summary>
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
                    string modelPath = System.IO.Path.Combine(baseDir, "best_model.pt");

                    if (System.IO.File.Exists(modelPath))
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

                        // 必须 await 动画方法，确保本步动作演示完再进行下一步 MCTS 搜索
                        GameResult result = await selfPlay.RunGameAsync(async b => await UpdateBoardWithAnimation(b));

                        buffer.AddRange(result.Examples);

                        Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 收集样本: {result.Examples.Count}");

                        if (buffer.Count >= 4096)
                        {
                            Log("[训练] 开始梯度下降...");
                            float loss = trainer.Train(buffer.Sample(4096), epochs: 15);
                            Dispatcher.Invoke(() => LossLabel.Text = loss.ToString("F4"));
                            ModelManager.SaveModel(model, modelPath);
                            Log($"[训练] 完成，Loss: {loss:F4}");
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