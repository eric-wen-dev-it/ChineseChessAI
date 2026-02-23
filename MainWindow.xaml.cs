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

        // 核心：引入独立于 UI 的业务大脑
        private TrainingOrchestrator _orchestrator;

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoardUI();
            this.Loaded += (s, e) => DrawBoardLines();

            _replayChannel = Channel.CreateBounded<List<Move>>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });

            // 初始化总指挥官，并订阅它的事件汇报
            _orchestrator = new TrainingOrchestrator();
            _orchestrator.OnLog += msg => AppendLog(msg);
            _orchestrator.OnLossUpdated += loss => Dispatcher.Invoke(() => LossLabel.Text = loss.ToString("F4"));
            _orchestrator.OnReplayRequested += moves => _replayChannel.Writer.TryWrite(moves);
            _orchestrator.OnError += err => Dispatcher.Invoke(() => MessageBox.Show(err, "错误", MessageBoxButton.OK, MessageBoxImage.Error));
            _orchestrator.OnTrainingStopped += () => Dispatcher.Invoke(() => StartBtn.IsEnabled = true);

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
                    AppendLog("[观战] 接收到最新战况，强制中断旧回放...");
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

            if (_replayChannel.Reader.Count == 0)
                await Task.Delay(3000);
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

        // ================= 3. 按钮交互模块 (事件委派给指挥官) =================

        private async void OnStartTrainingClick(object sender, RoutedEventArgs e)
        {
            if (_orchestrator.IsTraining)
                return;

            StartBtn.IsEnabled = false;
            // 通知总指挥官去开启线程干活，UI立即返回不阻塞
            await _orchestrator.StartSelfPlayAsync();
        }

        private void OnReplayLastClick(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("极速模式下，后台的最新对局已经自动推送至棋盘频道。");
        }

        private void OnLoadFileClick(object sender, RoutedEventArgs e)
        {
            if (_orchestrator.IsTraining)
            {
                MessageBox.Show("训练期间无法手动载入棋谱。", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "选择棋谱文件",
                Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*",
                InitialDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs")
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string fileContent = File.ReadAllText(openFileDialog.FileName);
                    string ucciRecord = "";
                    var lines = fileContent.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        if (line.StartsWith("棋谱:"))
                        {
                            ucciRecord = line.Substring(3).Trim();
                            break;
                        }
                    }

                    var moveList = new List<Move>();
                    var movesStr = ucciRecord.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (var mStr in movesStr)
                    {
                        var move = NotationConverter.UcciToMove(mStr);
                        if (move != null)
                            moveList.Add(move.Value);
                    }

                    _replayChannel.Writer.TryWrite(moveList);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"解析失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private async void OnLoadDatasetClick(object sender, RoutedEventArgs e)
        {
            if (_orchestrator.IsTraining)
            {
                MessageBox.Show("请先停止当前训练！", "提示", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "选择棋谱数据集",
                Filter = "支持的数据集 (*.csv;*.pgn;*.txt)|*.csv;*.pgn;*.txt|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                StartBtn.IsEnabled = false;
                AppendLog($"[系统] 准备处理文件: {System.IO.Path.GetFileName(openFileDialog.FileName)} ...");

                // 将繁重的文件解析任务外包给指挥官
                await _orchestrator.ProcessDatasetAsync(openFileDialog.FileName);
            }
        }
    }
}