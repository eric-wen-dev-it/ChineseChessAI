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

        private void UpdateBoard(Board board)
        {
            Dispatcher.Invoke(() =>
            {
                for (int i = 0; i < 90; i++)
                {
                    sbyte p = board.GetPiece(i);
                    _cellButtons[i].Content = GetPieceChar(p);
                    _cellButtons[i].Foreground = p > 0 ? Brushes.Red : Brushes.Black;

                    // --- 【核心修改：通过 Tag 标记起止位置以显示方框】 ---
                    if (board.LastMove != null)
                    {
                        if (i == board.LastMove.Value.From)
                        {
                            _cellButtons[i].Tag = "From"; // 触发 XAML 中的起始方框（如橙红色）
                        }
                        else if (i == board.LastMove.Value.To)
                        {
                            _cellButtons[i].Tag = "To";   // 触发 XAML 中的结束方框（如翠绿色）
                        }
                        else
                        {
                            _cellButtons[i].Tag = null;   // 清除其他位置的标记
                        }
                    }
                    else
                    {
                        _cellButtons[i].Tag = null;       // 局面重置时清除所有标记
                    }
                }

                // 自动更新棋谱序列
                MoveListLog.Text = board.GetMoveHistoryString();
            });
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

                    // 1. 初始化模型与环境
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

                    // 2. 初始化 ReplayBuffer 并加载磁盘旧样本
                    // 增大容量至 100,000 以支持更长期的经验回放
                    var buffer = new ReplayBuffer(100000);
                    Log("[系统] 正在扫描磁盘旧样本...");

                    // 调用您新添加的 LoadOldSamples 方法，从 data/self_play_data 恢复进度
                    buffer.LoadOldSamples();

                    var trainer = new Trainer(model);

                    // 3. 进入进化循环
                    for (int iter = 1; iter <= 10000; iter++)
                    {
                        Log($"\n--- [迭代: 第 {iter} 轮] ---");

                        // 执行自对弈：UpdateBoard 会处理 LastMove 的起始/结束位置方框高亮
                        GameResult result = await selfPlay.RunGameAsync(b => UpdateBoard(b));

                        // 将新样本加入缓存，并自动序列化到磁盘
                        buffer.AddRange(result.Examples);

                        // 日志输出详细的局终信息
                        Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 收集样本: {result.Examples.Count}");

                        // 4. 触发训练：当累积样本达到 4096 条时
                        if (buffer.Count >= 4096)
                        {
                            Log($"[训练] 正在从 {buffer.Count} 个样本中随机采样并开始梯度下降...");

                            // 进行 15 个 Epoch 的强化学习
                            float loss = trainer.Train(buffer.Sample(4096), epochs: 15);

                            // 更新 UI 上的平均 Loss 显示
                            Dispatcher.Invoke(() => LossLabel.Text = loss.ToString("F4"));

                            // 保存进化后的最佳模型并创建备份
                            ModelManager.SaveModel(model, modelPath);

                            Log($"[训练] 完成，当前 Loss: {loss:F4}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"[致命错误] {ex.Message}");
                    _isTraining = false; // 允许重新开始
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