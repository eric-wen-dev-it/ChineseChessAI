using ChineseChessAI.Core;
using ChineseChessAI.Training;
using ChineseChessAI.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Channels;
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
        private Channel<List<Move>> _replayChannel;

        private TrainingOrchestrator _orchestrator;

        // 【核心修复 1】新增标志位：阻断后台抢占焦点
        private volatile bool _isManualReplayActive = false;

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoardUI();
            this.Loaded += (s, e) => DrawBoardLines();

            _replayChannel = Channel.CreateBounded<List<Move>>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });

            _orchestrator = new TrainingOrchestrator();
            _orchestrator.OnLog += msg => AppendLog(msg);
            _orchestrator.OnLossUpdated += loss => Dispatcher.Invoke(() => LossLabel.Text = loss.ToString("F4"));

            // 【核心修复 2】后台完成对局时，只有在非手动模式下，才推送到前台
            _orchestrator.OnReplayRequested += moves =>
            {
                if (!_isManualReplayActive)
                {
                    _replayChannel.Writer.TryWrite(moves);
                }
            };

            _orchestrator.OnError += err => Dispatcher.Invoke(() => MessageBox.Show(err, "错误", MessageBoxButton.OK, MessageBoxImage.Error));
            _orchestrator.OnTrainingStopped += () => Dispatcher.Invoke(() => StartBtn.IsEnabled = true);

            DataContext = new TrainingConfig();
            _ = Task.Run(StartReplayLoopAsync);
        }

        // ================= 1. 纯 UI 渲染模块 =================

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

        private void AppendLog(string msg)
        {
            Dispatcher.Invoke(() =>
            {
                LogBox.AppendText($"{DateTime.Now:HH:mm:ss} - {msg}\n");
                LogBox.ScrollToEnd();
            });
        }

        // ================= 2. 动画播放模块 =================

        private async Task StartReplayLoopAsync()
        {
            try
            {
                await foreach (var gameMoves in _replayChannel.Reader.ReadAllAsync())
                {
                    AppendLog($"[观战] 开始播放对局动画，步数: {gameMoves.Count}");
                    await ReplayGameInternalAsync(gameMoves);
                }
            }
            catch (ChannelClosedException) { }
        }

        private async Task ReplayGameInternalAsync(List<Move> historyMoves)
        {
            Board uiBoard = new Board();
            uiBoard.Reset();

            Dispatcher.Invoke(() => RefreshBoardOnly(uiBoard));

            foreach (var move in historyMoves)
            {
                if (_replayChannel.Reader.Count > 0)
                {
                    AppendLog("[观战] 接收到最新指令，中断当前回放...");
                    break;
                }

                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                });

                await Task.Delay(400);

                if (_replayChannel.Reader.Count > 0)
                    break;

                uiBoard.Push(move.From, move.To);
                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                    _cellButtons[move.To].Tag = "To";
                });

                await Task.Delay(600);
            }

            // 【核心修复 3】如果完整播完了（没有被新的手动谱打断），解除锁定恢复后台推送
            if (_replayChannel.Reader.Count == 0)
            {
                await Task.Delay(3000);
                if (_isManualReplayActive)
                {
                    _isManualReplayActive = false;
                    AppendLog("[观战] 手动棋谱播放完毕，已恢复接收后台最新实战对局...");
                }
            }
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

        // ================= 3. 按钮交互模块 =================

        // 【安全修复】拦截 async void 的潜在崩溃
        private async void OnStartTrainingClick(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_orchestrator.IsTraining)
                    return;

                var config = (TrainingConfig)DataContext;
                if (!int.TryParse(config.MaxMoves, out int maxMoves) || maxMoves <= 0)
                {
                    MessageBox.Show("强制平局步数必须为正整数。", "参数错误", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }
                if (!int.TryParse(config.ExploreMoves, out int exploreMoves) || exploreMoves < 0)
                {
                    MessageBox.Show("高温探索步数必须为非负整数。", "参数错误", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }
                if (!float.TryParse(config.MaterialBias, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float materialBias) || materialBias < 0f)
                {
                    MessageBox.Show("破冰偏置必须为非负小数（如 0.05）。", "参数错误", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                StartBtn.IsEnabled = false;
                await _orchestrator.StartSelfPlayAsync(maxMoves, exploreMoves, materialBias);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"训练启动异常: {ex.Message}\n{ex.StackTrace}", "致命错误", MessageBoxButton.OK, MessageBoxImage.Error);
                StartBtn.IsEnabled = true;
            }
        }

        public class TrainingConfig
        {
            public string MaxMoves { get; set; } = "100";
            public string ExploreMoves { get; set; } = "40";
            public string MaterialBias { get; set; } = "0.6";
        }

        private void OnReplayLastClick(object sender, RoutedEventArgs e)
        {
            // 复用此按钮作为“取消手动模式，立刻恢复后台直播”的开关
            if (_isManualReplayActive)
            {
                _isManualReplayActive = false;
                AppendLog("[系统] 用户手动恢复了后台观战推送。");
                MessageBox.Show("已恢复接收后台最新对局！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            else
            {
                MessageBox.Show("极速模式下，后台的最新对局已经自动推送至棋盘频道。");
            }
        }

        private void OnLoadFileClick(object sender, RoutedEventArgs e)
        {
            // 【核心修复 4】彻底删除 _orchestrator.IsTraining 限制，实现随时载入
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "选择棋谱文件",
                Filter = "Text/PGN files (*.txt;*.pgn)|*.txt;*.pgn|All files (*.*)|*.*",
                InitialDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs")
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string fileContent = File.ReadAllText(openFileDialog.FileName);
                    string movesStr = "";

                    // 【核心修复 5】智能解析引擎：不仅兼容系统 Log，更兼容外部的中文/PGN格式
                    if (fileContent.Contains("棋谱:"))
                    {
                        var lines = fileContent.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                        foreach (var line in lines)
                        {
                            if (line.StartsWith("棋谱:"))
                            {
                                movesStr = line.Substring(3).Trim();
                                break;
                            }
                        }
                    }
                    else
                    {
                        // 剥离 PGN 头部标签，只保留着法区
                        movesStr = System.Text.RegularExpressions.Regex.Replace(fileContent, @"\[[^\]]*\]", "");
                        movesStr = System.Text.RegularExpressions.Regex.Replace(movesStr, @"\{[^}]*\}", "");
                        movesStr = System.Text.RegularExpressions.Regex.Replace(movesStr, @"\b\d+\.", "");
                        movesStr = movesStr.Replace("1-0", "").Replace("0-1", "").Replace("1/2-1/2", "").Replace("*", "");
                    }

                    var moveList = new List<Move>();
                    var rawMoves = movesStr.Split(new[] { ' ', '\n', '\r', '\t', ',' }, StringSplitOptions.RemoveEmptyEntries);

                    var tempBoard = new Board();
                    tempBoard.Reset();
                    var generator = new MoveGenerator();

                    foreach (var mStr in rawMoves)
                    {
                        // 接入万能转换器，支持 "炮二平五" 和 "h2e2" 混用
                        string ucci = NotationConverter.ConvertToUcci(tempBoard, mStr, generator);
                        if (!string.IsNullOrEmpty(ucci))
                        {
                            var move = NotationConverter.UcciToMove(ucci);
                            if (move != null)
                            {
                                moveList.Add(move.Value);
                                tempBoard.Push(move.Value.From, move.Value.To);
                            }
                        }
                    }

                    if (moveList.Count > 0)
                    {
                        // 【核心修复 6】阻断后台推送，专心为用户播放当前文件
                        _isManualReplayActive = true;
                        AppendLog($"[观战] 已成功导入并解析 {moveList.Count} 步棋谱，开启纯净回放模式...");
                        _replayChannel.Writer.TryWrite(moveList);
                    }
                    else
                    {
                        MessageBox.Show("未能解析出有效的走法，请检查文件内容是否包含正常的棋谱序列！", "解析失败", MessageBoxButton.OK, MessageBoxImage.Warning);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"解析失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        // 【安全修复】拦截 async void 的潜在崩溃
        private async void OnLoadDatasetClick(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_orchestrator.IsTraining)
                {
                    MessageBox.Show("当前正在训练中，请先停止当前训练再导入！", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                var openFileDialog = new Microsoft.Win32.OpenFileDialog
                {
                    Title = "选择巨型棋谱数据集",
                    Filter = "支持的数据集 (*.csv;*.pgn;*.txt)|*.csv;*.pgn;*.txt|All files (*.*)|*.*"
                };

                if (openFileDialog.ShowDialog() == true)
                {
                    var result = MessageBox.Show(
                        "是否在解析的同时进行训练？\n\n【是】解析 + 训练（较慢，占用 GPU）\n【否】仅解析存盘，不训练（快速，下次启动自动加载）",
                        "导入模式",
                        MessageBoxButton.YesNo,
                        MessageBoxImage.Question);

                    bool trainWhileParsing = result == MessageBoxResult.Yes;
                    StartBtn.IsEnabled = false;
                    AppendLog($"[系统] 准备吞噬处理巨型文件: {System.IO.Path.GetFileName(openFileDialog.FileName)} | 模式: {(trainWhileParsing ? "解析+训练" : "纯解析存盘")}");
                    await _orchestrator.ProcessDatasetAsync(openFileDialog.FileName, trainWhileParsing);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"导入数据集异常: {ex.Message}\n{ex.StackTrace}", "致命错误", MessageBoxButton.OK, MessageBoxImage.Error);
                StartBtn.IsEnabled = true;
            }
        }
    }
}