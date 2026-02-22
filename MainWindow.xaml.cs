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
                    Margin = new Thickness(2),
                    Tag = null // 初始化 Tag 为空
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

        /// <summary>
        /// 【核心修改】带分步动画的棋盘更新逻辑
        /// </summary>
        private async Task UpdateBoardWithAnimation(Board board)
        {
            var move = board.LastMove;

            // 如果是开局或重置，直接刷新
            if (move == null)
            {
                await Dispatcher.InvokeAsync(() => RefreshBoardOnly(board));
                return;
            }

            // 在 UI 线程执行分步演示
            await Dispatcher.InvokeAsync(async () =>
            {
                // 1. 恢复之前的物理状态（棋子在原位），仅绘制红色起始方框
                // 注意：这里需要先手动“撤销”视觉上的移动，展示起点
                sbyte movingPiece = board.GetPiece(move.Value.To);
                sbyte capturedPiece = board.GetPiece(move.Value.From); // 虽然已经是0，但在逻辑上它是起点

                // 强制将起点设为红框
                _cellButtons[move.Value.From].Tag = "From";
                _cellButtons[move.Value.To].Tag = null;

                // 等待让用户看清“起手”
                await Task.Delay(200);

                // 2. 正式移动棋子并绘制结束位置
                RefreshBoardOnly(board);

                // 确保起止点高亮同时存在
                _cellButtons[move.Value.From].Tag = "From";
                _cellButtons[move.Value.To].Tag = "To";
            });
        }

        /// <summary>
        /// 仅刷新棋子物理位置，不处理动画延迟
        /// </summary>
        private void RefreshBoardOnly(Board board)
        {
            for (int i = 0; i < 90; i++)
            {
                sbyte p = board.GetPiece(i);
                _cellButtons[i].Content = Board.GetPieceName(p);
                _cellButtons[i].Foreground = p > 0 ? Brushes.Red : Brushes.Black;
                _cellButtons[i].Tag = null; // 重置标记
            }
            MoveListLog.Text = board.GetMoveHistoryString();
        }

        private string GetPieceChar(sbyte p)
        {
            return Board.GetPieceName(p);
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

                    // 1. 初始化模型
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

                    // 2. 初始化缓存并加载旧样本
                    var buffer = new ReplayBuffer(100000);
                    Log("[系统] 正在扫描磁盘旧样本...");
                    buffer.LoadOldSamples();

                    var trainer = new Trainer(model);

                    // 3. 进入循环
                    for (int iter = 1; iter <= 10000; iter++)
                    {
                        Log($"\n--- [迭代: 第 {iter} 轮] ---");

                        // 【修改】回调改为异步函数以支持动画演示
                        GameResult result = await selfPlay.RunGameAsync(async b => await UpdateBoardWithAnimation(b));

                        buffer.AddRange(result.Examples);

                        Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 收集样本: {result.Examples.Count}");

                        if (buffer.Count >= 4096)
                        {
                            Log("[训练] 开始梯度下降...");
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