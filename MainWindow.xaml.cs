using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq; // 引入 Linq 以支持 Count()
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
            CheckGpuStatus(); // 确保构造函数调用此方法
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

        // 将 Board 状态绘制到 UI
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
            StartBtn.IsEnabled = false;

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

                    // 实例化模型（请确保 CChessNet.cs 内部已正确 RegisterModule）
                    var model = new CChessNet(numResBlocks: 10, numFilters: 128);

                    if (torch.cuda.is_available())
                    {
                        UpdateUI("[硬件] 检测到 GPU，正在启用 CUDA 加速");
                        model.to(DeviceType.CUDA);
                    }
                    else
                    {
                        UpdateUI("[硬件] 未检测到 GPU，使用 CPU 运行");
                        model.to(DeviceType.CPU);
                    }

                    string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "best_model.pt");
                    if (File.Exists(modelPath))
                    {
                        UpdateUI($"[加载] 发现权重文件，正在载入: {modelPath}");
                        ModelManager.LoadModel(model, modelPath);

                        // 【修复 A】加载后强制开启梯度，防止 backward 报错
                        foreach (var param in model.parameters())
                        {
                            param.requires_grad = true;
                        }
                        UpdateUI("[系统] 已强制激活加载模型的参数梯度记录");
                    }
                    else
                    {
                        // 新模型也强制开启梯度
                        foreach (var param in model.parameters())
                        {
                            param.requires_grad = true;
                        }
                        UpdateUI("[系统] 新模型已就绪并开启梯度记录");
                    }

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
                            // --- 阶段 A: 多线程并行自我对弈 ---
                            var gameTasks = new List<Task<List<TrainingExample>>>();
                            UpdateUI($"[对弈] 启动线程进行自我博弈...");

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
                                                moveCallback = (currentBoard) =>
                                                {
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

                            UpdateUI("[同步] 等待所有对弈完成...");
                            var allGameResults = await Task.WhenAll(gameTasks);
                            foreach (var gameData in allGameResults)
                            {
                                if (gameData != null)
                                    buffer.AddExamples(gameData);
                            }
                            UpdateUI($"[缓存] 当前 Buffer 总数: {buffer.Count}");

                            // --- 阶段 B: 神经网络训练 ---
                            if (buffer.Count >= 256)
                            {
                                UpdateUI("[训练] 满足样本量，开始梯度下降...");
                                double totalLoss = 0;
                                int trainSteps = 20;

                                for (int s = 0; s < trainSteps; s++)
                                {
                                    try
                                    {
                                        using (var batchScope = torch.NewDisposeScope())
                                        {
                                            var (states, policies, values) = buffer.Sample(32);

                                            // 【修复 B】训练前最后的诊断与自愈
                                            var paramList = model.parameters().ToList();
                                            if (paramList.Count == 0)
                                            {
                                                UpdateUI("[致命错误] 模型参数数量为0，RegisterComponents 注册失败！");
                                                break;
                                            }
                                            if (!paramList[0].requires_grad)
                                            {
                                                foreach (var p in paramList)
                                                    p.requires_grad = true;
                                            }

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

                            UpdateUI("[IO] 正在保存模型权重...");
                            ModelManager.SaveModel(model, modelPath);

                            iteration++;
                            UpdateUI($"[清理] 第 {iteration - 1} 轮结束，显存已释放");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                UpdateUI($"[全局错误] 进化循环崩溃: {ex.Message}");
                Debug.WriteLine($"[全局错误] 崩溃堆栈: {ex.StackTrace}");
            }
            finally
            {
                _isTraining = false;
                UpdateUI("[系统] 训练流程已完全停止。");
                Dispatcher.Invoke(() => StartBtn.IsEnabled = true);
            }
        }

        private void UpdateUI(string message)
        {
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