using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using TorchSharp;
using static TorchSharp.torch;

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
            UpdateUI("[系统] 启动进化训练流程...");
            Debug.WriteLine("[系统] 启动进化训练流程...");

            try
            {
                using (var outerScope = torch.NewDisposeScope())
                {
                    UpdateUI("[初始化] 正在构建 CChessNet 模型...");
                    Debug.WriteLine("[初始化] 正在构建 CChessNet 模型...");

                    var model = new CChessNet(numResBlocks: 10, numFilters: 128);

                    if (torch.cuda.is_available())
                    {
                        UpdateUI("[硬件] 检测到 GPU，正在启用 CUDA 加速");
                        Debug.WriteLine("[硬件] 检测到 GPU，正在启用 CUDA 加速");
                        //model.to(DeviceType.CUDA);
                        model.to(DeviceType.CUDA, ScalarType.Float16);
                    }
                    else
                    {
                        UpdateUI("[硬件] 未检测到 GPU，使用 CPU 运行");
                        Debug.WriteLine("[硬件] 未检测到 GPU，使用 CPU 运行");
                        model.to(DeviceType.CPU);
                    }

                    string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
                    if (File.Exists(modelPath))
                    {
                        UpdateUI($"[加载] 发现权重文件，正在载入: {modelPath}");
                        Debug.WriteLine($"[加载] 发现权重文件，正在载入: {modelPath}");
                        ModelManager.LoadModel(model, modelPath);
                    }

                    var trainer = new Trainer(model);
                    var buffer = new ReplayBuffer(capacity: 50000);
                    var engine = new MCTSEngine(model);
                    var selfPlay = new SelfPlay(engine);

                    int iteration = 1;
                    while (_isTraining)
                    {
                        UpdateUI($"\n--- [迭代进度: 第 {iteration} 轮] ---");
                        Debug.WriteLine($"\n--- [迭代进度: 第 {iteration} 轮] ---");

                        using (var iterScope = torch.NewDisposeScope())
                        {
                            // --- 阶段 A: 多线程并行自我对弈 ---
                            var gameTasks = new List<Task<List<TrainingExample>>>();
                            UpdateUI($"[对弈] 启动 3 个线程进行自我博弈...");
                            Debug.WriteLine($"[对弈] 启动 3 个线程进行自我博弈...");

                            for (int g = 1; g <= 3; g++)
                            {
                                int gameId = g;
                                gameTasks.Add(Task.Run(() =>
                                {
                                    using (var threadScope = torch.NewDisposeScope())
                                    {
                                        try
                                        {
                                            Debug.WriteLine($"[线程调试] 游戏线程 {gameId} 开始运行...");
                                            Action<Board>? moveCallback = null;
                                            if (gameId == 1)
                                            {
                                                int moveStep = 0;
                                                moveCallback = (currentBoard) =>
                                                {
                                                    moveStep++;
                                                    if (moveStep % 10 == 0)
                                                    {
                                                        var boardSnapshot = currentBoard.GetState();
                                                        Dispatcher.BeginInvoke(new Action(() => DrawBoard(boardSnapshot)));
                                                    }
                                                };
                                            }
                                            return selfPlay.RunGame(moveCallback);
                                        }
                                        catch (Exception ex)
                                        {
                                            UpdateUI($"[致命] 线程 {gameId} 崩溃: {ex.Message}");
                                            Debug.WriteLine($"[致命] 线程 {gameId} 崩溃: {ex.Message}\n{ex.StackTrace}");
                                            return new List<TrainingExample>();
                                        }
                                    }
                                }));
                            }

                            UpdateUI("[同步] 等待所有对弈完成...");
                            Debug.WriteLine("[同步] 等待所有对弈完成...");
                            var allGameResults = await Task.WhenAll(gameTasks);
                            foreach (var gameData in allGameResults)
                            {
                                // 此时 gameData 的类型是 List<TrainingExample>
                                // 调用的 buffer.AddExamples 也是接收 List<TrainingExample>
                                if (gameData != null)
                                    buffer.AddExamples(gameData);
                            }
                            UpdateUI($"[缓存] 对弈结束，当前 Buffer 总数: {buffer.Count}");
                            Debug.WriteLine($"[缓存] 对弈结束，当前 Buffer 总数: {buffer.Count}");

                            // --- 阶段 B: 神经网络训练 ---
                            if (buffer.Count >= 256)
                            {
                                UpdateUI("[训练] 满足样本量，开始梯度下降...");
                                Debug.WriteLine("[训练] 满足样本量，开始梯度下降...");
                                double totalLoss = 0;
                                int trainSteps = 20;

                                try
                                {
                                    for (int s = 0; s < trainSteps; s++)
                                    {
                                        try
                                        {
                                            // 这是最核心的：确保每一批次训练的中间 Tensor 都在此销毁
                                            using (var batchScope = torch.NewDisposeScope())
                                            {
                                                Debug.WriteLine($"[训练调试] Step {s + 1}/{trainSteps} 准备采样...");
                                                var (states, policies, values) = buffer.Sample(32);

                                                Debug.WriteLine($"[训练调试] Step {s + 1} 采样完成，进入 TrainStep...");
                                                // 请确保 states, policies, values 已经 model.to(Device)
                                                double loss = trainer.TrainStep(states, policies, values);

                                                totalLoss += loss;
                                                Debug.WriteLine($"[训练调试] Step {s + 1} 成功完成，Loss: {loss:F4}");
                                            }
                                        }
                                        catch (Exception ex)
                                        {
                                            UpdateUI($"[训练警告] Step {s + 1} 失败: {ex.Message}");
                                        }

                                        // 如果你担心内存堆积，这里用 GC 替代
                                        //if (s % 5 == 0)
                                        //{
                                        //    GC.Collect();
                                        //    GC.WaitForPendingFinalizers();
                                        //}
                                    }
                                    UpdateUI($"[训练] 完毕，平均 Loss: {totalLoss / trainSteps:F4}");
                                    Debug.WriteLine($"[训练] 完毕，平均 Loss: {totalLoss / trainSteps:F4}");
                                }
                                catch (Exception ex)
                                {
                                    UpdateUI($"[异常] 训练循环内崩溃: {ex.Message}");
                                    Debug.WriteLine($"[异常] 训练循环内崩溃: {ex.Message}\n{ex.StackTrace}");
                                    break;
                                }
                            }

                            UpdateUI("[IO] 正在持久化当前模型权重...");
                            Debug.WriteLine("[IO] 正在持久化当前模型权重...");
                            ModelManager.SaveModel(model, modelPath);

                            iteration++;
                            UpdateUI($"[清理] 结束第 {iteration - 1} 轮，释放临时显存");
                            Debug.WriteLine($"[清理] 结束第 {iteration - 1} 轮，释放临时显存");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                UpdateUI($"[全局错误] 进化循环崩溃: {ex.Message}");
                Debug.WriteLine($"[全局错误] 进化循环崩溃: {ex.Message}\n{ex.StackTrace}");
            }
            finally
            {
                _isTraining = false;
                UpdateUI("[系统] 训练流程已完全停止。");
                Debug.WriteLine("[系统] 训练流程已完全停止。");
            }
        }

        // 统一的日志输出方法，修复 LogText 缺失问题
        private void UpdateUI(string message)
        {
            // 修复：确保在 UI 线程执行
            Dispatcher.Invoke(() =>
            {
                if (LogBox != null)
                {
                    LogBox.AppendText($"{DateTime.Now:HH:mm:ss} - {message}\n");
                    LogBox.ScrollToEnd();
                }
            });
        }
    }
}