using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using TorchSharp;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using ChineseChessAI.Core;

namespace ChineseChessAI
{
    public partial class MainWindow : Window
    {
        private bool _isTraining = false;

        public MainWindow()
        {
            InitializeComponent();
            CheckGpuStatus(); // 修复：确保构造函数调用此方法
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

        private void RunEvolutionLoop()
        {
            try
            {
                // 使用 DisposeScope 自动管理非托管 Tensor 内存
                using (var outerScope = torch.NewDisposeScope())
                {
                    var model = new CChessNet(numResBlocks: 10, numFilters: 128);

                    // 如果 model.to 支持直接接收枚举
                    if (torch.cuda.is_available())
                    {
                        model.to(DeviceType.CUDA);
                    }
                    else
                    {
                        model.to(DeviceType.CPU);
                    }

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
                        // 每一轮迭代也开启一个内层作用域，防止内存累积
                        using (var iterScope = torch.NewDisposeScope())
                        {
                            UpdateUI($"\n--- 迭代第 {iteration} 轮 ---");

                            // 阶段 A: 自我对弈
                            for (int g = 1; g <= 3; g++)
                            {
                                var gameData = selfPlay.RunGame();
                                buffer.AddExamples(gameData);
                                UpdateUI($"[对弈] 游戏 {g}/3 完成。");
                            }

                            // 阶段 B: 训练
                            if (buffer.Count >= 256)
                            {
                                double totalLoss = 0;
                                for (int s = 0; s < 20; s++)
                                {
                                    var (states, policies, values) = buffer.Sample(32);
                                    totalLoss += trainer.TrainStep(states, policies, values);
                                }
                                UpdateUI($"[训练] 平均 Loss: {totalLoss / 20:F4}");
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