using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace ChineseChessAI
{
    public partial class MainWindow : Window
    {
        private Button[] _cellButtons = new Button[90];
        private bool _isTraining = false;

        // 【新增】观战状态锁，防止多个动画同时抢占 UI
        private bool _isReplaying = false;

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
            double w = ChessLinesCanvas.ActualWidth, h = ChessLinesCanvas.ActualHeight;
            double stepX = w / 9, stepY = h / 10;

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
            ChessLinesCanvas.Children.Add(new Line { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Stroke = Brushes.Black, StrokeThickness = 1.2 });
        }

        /// <summary>
        /// 【全新独立复盘任务】
        /// 接收完整的一局棋走法，使用一个完全独立的 Board 进行慢动作演示。
        /// 不会干扰后台的高速训练。
        /// </summary>
        private async Task ReplayGameAsync(List<Move> historyMoves)
        {
            _isReplaying = true;

            // 1. 创建 UI 专用的棋盘，绝对不会和训练线程冲突
            Board uiBoard = new Board();
            uiBoard.Reset();

            Dispatcher.Invoke(() => RefreshBoardOnly(uiBoard));

            // 2. 慢慢回放这局棋
            foreach (var move in historyMoves)
            {
                // 先拿到被吃的子（如果有的话）
                sbyte capturedPiece = uiBoard.GetPiece(move.To);

                // 阶段 1：起手（不改变底层数据，只做 UI 视觉欺骗）
                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    sbyte movingPiece = uiBoard.GetPiece(move.From);

                    _cellButtons[move.From].Foreground = movingPiece > 0 ? Brushes.Red : Brushes.Black;
                    _cellButtons[move.From].Tag = "From";

                    if (capturedPiece != 0)
                    {
                        _cellButtons[move.To].Content = Board.GetPieceName(capturedPiece);
                        _cellButtons[move.To].Foreground = capturedPiece > 0 ? Brushes.Red : Brushes.Black;
                    }
                });

                // 悬停动画等待
                await Task.Delay(600);

                // 阶段 2：正式落子
                uiBoard.Push(move.From, move.To); // 更新底层 UI 棋盘数据
                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                    _cellButtons[move.To].Tag = "To";
                });
            }

            // 终局停留 2 秒供观察结果
            await Task.Delay(2000);
            _isReplaying = false; // 释放锁，允许播放下一局
        }

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
                    Log("=== 进化循环已启动 (极速模式) ===");
                    var model = new CChessNet();
                    string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                    string modelPath = Path.Combine(baseDir, "best_model.pt");

                    if (File.Exists(modelPath))
                    {
                        model.load(modelPath);
                        Log("[系统] 已加载现有模型权重。");
                    }

                    var engine = new MCTSEngine(model, batchSize: 512);
                    var selfPlay = new SelfPlay(engine);
                    var buffer = new ReplayBuffer(100000);
                    buffer.LoadOldSamples();

                    var trainer = new Trainer(model);

                    for (int iter = 1; iter <= 10000; iter++)
                    {
                        Log($"\n--- [迭代: 第 {iter} 轮] 正在后台极速对弈... ---");

                        // 【核心改动】：传入 null，关闭训练过程中的同步动画回调，让它全速奔跑！
                        GameResult result = await selfPlay.RunGameAsync(null);

                        // 【核心改动】：对弈瞬间完成，如果前台处于空闲状态，就把这局丢给前台慢慢播放 (Fire and Forget)
                        if (!_isReplaying && result.MoveHistory != null && result.MoveHistory.Count > 0)
                        {
                            Log($"[观战] 提取本局棋谱投射至 UI 进行慢动作演示...");
                            _ = ReplayGameAsync(result.MoveHistory);
                        }

                        // 保存棋谱等常规操作 (依然在后台飞速进行)
                        string moveStr = string.Join(" ", result.MoveHistory.Select(m => m.ToString()));
                        SaveMoveListToFile(moveStr, result.ResultStr, result.EndReason);

                        if (result.MoveCount > 10)
                        {
                            buffer.AddRange(result.Examples);
                            Log($"[对弈] 结束 ({result.EndReason}) | 结果: {result.ResultStr} | 步数: {result.MoveCount} | 样本已存入");
                        }
                        else
                        {
                            Log($"[对弈] 警告: 步数过短({result.MoveCount})，视为无效博弈，样本已舍弃。");
                        }

                        if (buffer.Count >= 4096)
                        {
                            Log($"[训练] 开始梯度下降... 当前学习率: {trainer.GetCurrentLR():F6}");
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

        private void SaveMoveListToFile(string moveList, string result, string reason)
        {
            try
            {
                string logDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs");
                if (!Directory.Exists(logDir))
                    Directory.CreateDirectory(logDir);

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string filePath = Path.Combine(logDir, $"game_{timestamp}.txt");

                string content = $"时间: {DateTime.Now}\n" +
                                 $"结果: {result}\n" +
                                 $"原因: {reason}\n" +
                                 $"棋谱: {moveList}\n" +
                                 new string('-', 40) + "\n";
                File.WriteAllText(filePath, content);
            }
            catch (Exception) { /* 忽略日志写入错误 */ }
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