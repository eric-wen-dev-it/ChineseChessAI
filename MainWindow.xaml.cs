using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes; // 必须引入用于 Line 绘制
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI
{
    public partial class MainWindow : Window
    {
        private bool _isTraining = false;

        public MainWindow()
        {
            InitializeComponent();

            // 订阅尺寸改变事件，确保拉伸窗口时正式棋盘线条自动重绘
            // 假设您在 XAML 中添加了一个名为 ChessLinesCanvas 的 Canvas
            this.Loaded += (s, e) => {
                if (FindName("ChessLinesCanvas") is Canvas canvas)
                {
                    canvas.SizeChanged += (ss, ee) => DrawChessBoardLines();
                    DrawChessBoardLines();
                }
            };

            InitBoardUi();
            CheckGpuStatus();
        }

        /// <summary>
        /// 使用绘图函数在 Canvas 上绘制正式棋盘线条
        /// </summary>
        private void DrawChessBoardLines()
        {
            if (!(FindName("ChessLinesCanvas") is Canvas canvas))
                return;

            canvas.Children.Clear();
            double w = canvas.ActualWidth;
            double h = canvas.ActualHeight;
            if (w <= 0 || h <= 0)
                return;

            double cellW = w / 9;
            double cellH = h / 10;
            Brush lineBrush = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#3E2723")); // 深木色线条
            double thickness = 1.5;

            // 1. 绘制纵向线 (垂直线)
            for (int c = 0; c < 9; c++)
            {
                double x = cellW * c + cellW / 2;
                if (c == 0 || c == 8) // 左右边框贯通
                {
                    DrawLine(canvas, x, cellH / 2, x, h - cellH / 2, lineBrush, thickness);
                }
                else // 中间线在楚河汉界断开
                {
                    DrawLine(canvas, x, cellH / 2, x, cellH * 4.5, lineBrush, thickness);
                    DrawLine(canvas, x, cellH * 5.5, x, h - cellH / 2, lineBrush, thickness);
                }
            }

            // 2. 绘制横向线 (水平线)
            for (int r = 0; r < 10; r++)
            {
                double y = cellH * r + cellH / 2;
                DrawLine(canvas, cellW / 2, y, w - cellW / 2, y, lineBrush, thickness);
            }

            // 3. 绘制九宫格斜线 (X形)
            // 上方九宫 (黑方)
            DrawLine(canvas, cellW * 3.5, cellH / 2, cellW * 5.5, cellH * 2.5, lineBrush, thickness);
            DrawLine(canvas, cellW * 5.5, cellH / 2, cellW * 3.5, cellH * 2.5, lineBrush, thickness);
            // 下方九宫 (红方)
            DrawLine(canvas, cellW * 3.5, h - cellH / 2, cellW * 5.5, h - cellH * 2.5, lineBrush, thickness);
            DrawLine(canvas, cellW * 5.5, h - cellH / 2, cellW * 3.5, h - cellH * 2.5, lineBrush, thickness);

            // 4. 添加“楚河 汉界”文字
            AddRiverText(canvas, "楚 河", cellW * 2.5, cellH * 5, lineBrush);
            AddRiverText(canvas, "汉 界", cellW * 6.5, cellH * 5, lineBrush);
        }

        private void DrawLine(Canvas canvas, double x1, double y1, double x2, double y2, Brush brush, double thickness)
        {
            canvas.Children.Add(new Line { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Stroke = brush, StrokeThickness = thickness });
        }

        private void AddRiverText(Canvas canvas, string text, double x, double y, Brush brush)
        {
            var txt = new TextBlock { Text = text, FontSize = 24, FontWeight = FontWeights.Bold, Foreground = brush, FontFamily = new FontFamily("楷体") };
            Canvas.SetLeft(txt, x - 30);
            Canvas.SetTop(txt, y - 15);
            canvas.Children.Add(txt);
        }

        private void InitBoardUi()
        {
            ChessBoardGrid.Children.Clear();
            for (int i = 0; i < 90; i++)
            {
                // 棋子显示容器 (圆形)
                var pieceBorder = new Border
                {
                    Width = 44,
                    Height = 44,
                    CornerRadius = new CornerRadius(22),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    Background = Brushes.Transparent,
                    BorderThickness = new Thickness(0)
                };

                var txt = new TextBlock
                {
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    FontSize = 22,
                    FontWeight = FontWeights.Bold,
                    FontFamily = new FontFamily("Microsoft YaHei")
                };

                pieceBorder.Child = txt;
                ChessBoardGrid.Children.Add(pieceBorder);
            }
        }

        private void DrawBoard(sbyte[] pieces)
        {
            for (int i = 0; i < 90; i++)
            {
                var border = (Border)ChessBoardGrid.Children[i];
                var txt = (TextBlock)border.Child;
                sbyte piece = pieces[i];

                if (piece == 0)
                {
                    txt.Text = "";
                    border.Background = Brushes.Transparent;
                    border.BorderThickness = new Thickness(0);
                    border.Effect = null;
                    continue;
                }

                string[] names = { "", "帅", "仕", "相", "马", "车", "炮", "兵" };
                txt.Text = names[Math.Abs(piece)];

                // 立体棋子样式
                border.Background = Brushes.White;
                border.BorderThickness = new Thickness(2);
                border.Effect = new System.Windows.Media.Effects.DropShadowEffect { BlurRadius = 4, Opacity = 0.3 };

                if (piece > 0)
                { // 红方
                    border.BorderBrush = Brushes.Red;
                    txt.Foreground = Brushes.Red;
                }
                else
                { // 黑方
                    border.BorderBrush = Brushes.Black;
                    txt.Foreground = Brushes.Black;
                }
            }
        }

        private void CheckGpuStatus()
        {
            if (torch.cuda.is_available())
            {
                int deviceCount = (int)torch.cuda.device_count();
                UpdateUI($"[环境] CUDA 就绪！检测到 {deviceCount} 个 GPU 设备。");
            }
            else
            {
                UpdateUI("[环境] 警告：未发现 GPU，将使用 CPU 运行。");
            }
        }

        private async void OnStartTrainingClick(object sender, RoutedEventArgs e)
        {
            if (_isTraining)
                return;
            _isTraining = true;
            StartBtn.IsEnabled = false;

            UpdateUI("=== 进化循环已启动 ===");
            await Task.Run(() => RunEvolutionLoop());
        }

        private async void RunEvolutionLoop()
        {
            UpdateUI("[系统] 启动进化训练流程...");
            try
            {
                using (var outerScope = torch.NewDisposeScope())
                {
                    UpdateUI("[初始化] 正在构建 CChessNet 模型...");
                    var model = new CChessNet(numResBlocks: 10, numFilters: 128);

                    // 1. 先进行设备移动
                    if (torch.cuda.is_available())
                    {
                        model.to(DeviceType.CUDA);
                    }
                    else
                    {
                        model.to(DeviceType.CPU);
                    }

                    // 2. 加载权重
                    string modelPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
                    if (File.Exists(modelPath))
                    {
                        UpdateUI($"[加载] 载入权重: {modelPath}");
                        ModelManager.LoadModel(model, modelPath);
                    }

                    // 3. 【核心修正】在设备移动后开启梯度
                    var parameters = model.parameters().ToList();
                    foreach (var param in parameters)
                    {
                        param.requires_grad = true;
                    }
                    UpdateUI($"[系统] 模型参数已激活，参数总数: {parameters.Count}");

                    if (parameters.Count == 0)
                    {
                        UpdateUI("[致命错误] 无法识别模型参数！请检查 CChessNet.cs。");
                        return;
                    }

                    // 4. 最后创建 Trainer
                    var trainer = new Trainer(model);
                    var buffer = new ReplayBuffer(capacity: 50000);
                    var engine = new MCTSEngine(model);
                    var selfPlay = new SelfPlay(engine);

                    int iteration = 1;
                    while (_isTraining)
                    {
                        UpdateUI($"\n--- [迭代进度: 第 {iteration} 轮] ---");
                        using (var iterScope = torch.NewDisposeScope())
                        {
                            // 阶段 A: 自我对弈
                            var gameTasks = new List<Task<List<TrainingExample>>>();
                            UpdateUI($"[对弈] 启动线程中...");

                            for (int g = 1; g <= 3; g++)
                            {
                                int gameId = g;
                                gameTasks.Add(Task.Run(async () =>
                                {
                                    using (var threadScope = torch.NewDisposeScope())
                                    {
                                        try
                                        {
                                            Action<Board>? moveCallback = null;
                                            if (gameId == 1)
                                            {
                                                moveCallback = (currentBoard) => {
                                                    var boardSnapshot = currentBoard.GetState();
                                                    Dispatcher.BeginInvoke(new Action(() => DrawBoard(boardSnapshot)));
                                                };
                                            }
                                            return await selfPlay.RunGameAsync(moveCallback);
                                        }
                                        catch (Exception ex)
                                        {
                                            UpdateUI($"[致命] 线程 {gameId} 崩溃: {ex.Message}");
                                            return new List<TrainingExample>();
                                        }
                                    }
                                }));
                            }

                            var allGameResults = await Task.WhenAll(gameTasks);
                            foreach (var gameData in allGameResults)
                                if (gameData != null)
                                    buffer.AddExamples(gameData);

                            UpdateUI($"[缓存] Buffer 总数: {buffer.Count}");

                            // 阶段 B: 训练
                            if (buffer.Count >= 256)
                            {
                                UpdateUI("[训练] 开始梯度下降...");
                                double totalLoss = 0;
                                int trainSteps = 20;

                                for (int s = 0; s < trainSteps; s++)
                                {
                                    try
                                    {
                                        using (var batchScope = torch.NewDisposeScope())
                                        {
                                            var (states, policies, values) = buffer.Sample(32);
                                            double loss = trainer.TrainStep(states, policies, values);
                                            totalLoss += loss;
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        UpdateUI($"[训练警告] Step {s + 1} 失败: {ex.Message}");
                                    }
                                }
                                UpdateUI($"[训练] 完毕，平均 Loss: {totalLoss / trainSteps:F4}");
                            }

                            UpdateUI("[IO] 保存模型权重...");
                            ModelManager.SaveModel(model, modelPath);
                            iteration++;
                            UpdateUI($"[清理] 结束第 {iteration - 1} 轮");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                UpdateUI($"[全局错误] {ex.Message}");
            }
            finally
            {
                _isTraining = false;
                UpdateUI("[系统] 流程已停止。");
                Dispatcher.Invoke(() => StartBtn.IsEnabled = true);
            }
        }

        private void UpdateUI(string message)
        {
            Dispatcher.Invoke(() => {
                if (LogBox != null)
                {
                    LogBox.AppendText($"{DateTime.Now:HH:mm:ss} - {message}\n");
                    LogBox.ScrollToEnd();
                }
            });
        }
    }
}