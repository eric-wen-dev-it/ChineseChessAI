using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using TorchSharp;
using System.Windows.Media;

namespace ChineseChessAI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private bool _isTraining = false;

        public MainWindow()
        {
            InitializeComponent();
            InitBoardUi();
            CheckGpuStatus(); // 修复：确保构造函数调用此方法
        }

        private void InitBoardUi()
        {
            ChessBoardGrid.Children.Clear();
            for (int i = 0; i < 90; i++)
            {
                var border = new Border { BorderBrush = Brushes.Black, BorderThickness = new Thickness(0.5) };
                var txt = new TextBlock
                {
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    FontSize = 18,
                    FontWeight = FontWeights.Bold
                };
                border.Child = txt;
                ChessBoardGrid.Children.Add(border);
            }
        }

        // 核心：将 Board 状态绘制到 UI
        // 修改 DrawBoard 签名，直接接收棋子数据副本
        private void DrawBoard(sbyte[] pieces)
        {
            // 注意：这里不再需要 Dispatcher.Invoke，因为调用方已经使用了 BeginInvoke
            for (int i = 0; i < 90; i++)
            {
                var border = (Border)ChessBoardGrid.Children[i];
                var txt = (TextBlock)border.Child;
                sbyte piece = pieces[i]; // 使用传入的副本数据

                if (piece == 0)
                {
                    txt.Text = "";
                    continue;
                }
                string[] names = { "", "帅", "仕", "相", "马", "车", "炮", "兵" };
                txt.Text = names[Math.Abs(piece)];
                txt.Foreground = piece > 0 ? Brushes.Red : Brushes.Black;
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
            StartBtn.IsEnabled = false; // 修复：禁用 XAML 中定义的 StartBtn

            UpdateUI("=== 进化循环已启动 ===");

            // 启动后台线程执行死循环，防止 UI 卡死
            await Task.Run(() => RunEvolutionLoop());
        }

        private async void RunEvolutionLoop()
        {
            try
            {
                // outerScope 确保模型等对象在训练停止时彻底释放显存
                using (var outerScope = torch.NewDisposeScope())
                {
                    var model = new CChessNet(numResBlocks: 10, numFilters: 128);

                    if (torch.cuda.is_available())
                        model.to(DeviceType.CUDA); // 移动到 GPU
                    else
                        model.to(DeviceType.CPU);

                    string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
                    if (File.Exists(modelPath))
                        ModelManager.LoadModel(model, modelPath);

                    var trainer = new Trainer(model);
                    var buffer = new ReplayBuffer(capacity: 50000);
                    var engine = new MCTSEngine(model);
                    var selfPlay = new SelfPlay(engine);

                    int iteration = 1;
                    while (_isTraining)
                    {
                        // iterScope 管理每一轮迭代产生的临时 Tensor 废料
                        using (var iterScope = torch.NewDisposeScope())
                        {
                            UpdateUI($"\n--- 迭代第 {iteration} 轮 ---");

                            // --- 阶段 A: 多线程并行自我对弈 ---
                            var gameTasks = new List<Task<List<TrainingExample>>>();

                            for (int g = 1; g <= 3; g++)
                            {
                                int gameId = g;
                                gameTasks.Add(Task.Run(() =>
                                {
                                    Action<Board>? moveCallback = null;
                                    if (gameId == 1)
                                    {
                                        int moveStep = 0;
                                        moveCallback = (currentBoard) =>
                                        {
                                            moveStep++;
                                            if (moveStep % 10 == 0)
                                            {
                                                // 关键修复：立即提取纯 C# 数组快照，防止异步访问已释放的内存
                                                var boardSnapshot = currentBoard.GetState();
                                                Dispatcher.BeginInvoke(new Action(() => DrawBoard(boardSnapshot)));
                                            }
                                        };
                                    }
                                    return selfPlay.RunGame(moveCallback);
                                }));
                            }

                            // 等待并行对弈结束
                            var allGameResults = await Task.WhenAll(gameTasks);
                            foreach (var gameData in allGameResults)
                            {
                                buffer.AddExamples(gameData);
                            }
                            UpdateUI($"[对弈] 本轮完成，当前缓存条数: {buffer.Count}");

                            // --- 阶段 B: 神经网络训练 ---
                            if (buffer.Count >= 256)
                            {
                                double totalLoss = 0;
                                int trainSteps = 20;
                                for (int s = 0; s < trainSteps; s++)
                                {
                                    var (states, policies, values) = buffer.Sample(32);
                                    totalLoss += trainer.TrainStep(states, policies, values);
                                }
                                UpdateUI($"[训练] 平均 Loss: {totalLoss / trainSteps:F4}");
                            }

                            ModelManager.SaveModel(model, modelPath);
                            iteration++;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                UpdateUI($"[错误] 运行时崩溃: {ex.Message}");
            }
        }

        // 统一的日志输出方法，修复 LogText 缺失问题
        private void UpdateUI(string message)
        {
            // 修复：确保在 UI 线程执行
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