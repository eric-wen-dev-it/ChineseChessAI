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
using System.Linq;

namespace ChineseChessAI
{
    public partial class MainWindow : Window
    {
        private Button[] _cellButtons = new Button[90];
        private Channel<(List<Move> moves, int limit, int gameId, string result)> _replayChannel;
        private TrainingOrchestrator _orchestrator;
        private volatile bool _isManualReplayActive = false;

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoardUI();
            this.Loaded += (s, e) => DrawBoardLines();

            _replayChannel = Channel.CreateBounded<(List<Move>, int, int, string)>(new BoundedChannelOptions(10)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });

            _orchestrator = new TrainingOrchestrator();
            _orchestrator.OnLog += msg => AppendLog(msg);
            _orchestrator.OnReplayRequested += (moves, limit, gameId, result) =>
            {
                if (!_isManualReplayActive)
                {
                    _replayChannel.Writer.TryWrite((moves, limit, gameId, result));
                }
            };

            _orchestrator.OnAuditFailureRequested += (history, illegalMove, reason) => 
            {
                // 当审计失败时，将该局面推送到演示队列
                _replayChannel.Writer.TryWrite((history.Concat(new[] { illegalMove }).ToList(), 0, -1, $"审计失败: {reason}"));
            };

            _orchestrator.OnError += err => Dispatcher.Invoke(() => MessageBox.Show(err, "错误", MessageBoxButton.OK, MessageBoxImage.Error));
            _orchestrator.OnTrainingStopped += () => Dispatcher.Invoke(() =>
            {
                StartLeagueBtn.IsEnabled = true;
            });

            DataContext = new TrainingConfig();
            _ = Task.Run(StartReplayLoopAsync);
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
            if (ChessLinesCanvas == null) return;
            ChessLinesCanvas.Children.Clear();
            double w = ChessLinesCanvas.ActualWidth, h = ChessLinesCanvas.ActualHeight;
            double stepX = w / 9, stepY = h / 10;

            var gridPen = new SolidColorBrush(Color.FromRgb(62, 39, 35));
            for (int i = 0; i < 10; i++)
                DrawLine(stepX / 2, i * stepY + stepY / 2, w - stepX / 2, i * stepY + stepY / 2, gridPen, 1.5);
            for (int i = 0; i < 9; i++)
            {
                if (i == 0 || i == 8)
                    DrawLine(i * stepX + stepX / 2, stepY / 2, i * stepX + stepX / 2, h - stepY / 2, gridPen, 1.5);
                else
                {
                    DrawLine(i * stepX + stepX / 2, stepY / 2, i * stepX + stepX / 2, 4 * stepY + stepY / 2, gridPen, 1.5);
                    DrawLine(i * stepX + stepX / 2, 5 * stepY + stepY / 2, i * stepX + stepX / 2, h - stepY / 2, gridPen, 1.5);
                }
            }

            DrawLine(3 * stepX + stepX / 2, stepY / 2, 5 * stepX + stepX / 2, 2 * stepY + stepY / 2, gridPen, 1.2);
            DrawLine(5 * stepX + stepX / 2, stepY / 2, 3 * stepX + stepX / 2, 2 * stepY + stepY / 2, gridPen, 1.2);
            DrawLine(3 * stepX + stepX / 2, 7 * stepY + stepY / 2, 5 * stepX + stepX / 2, 9 * stepY + stepY / 2, gridPen, 1.2);
            DrawLine(5 * stepX + stepX / 2, 7 * stepY + stepY / 2, 3 * stepX + stepX / 2, 9 * stepY + stepY / 2, gridPen, 1.2);

            DrawStarMarker(1, 2, stepX, stepY); DrawStarMarker(7, 2, stepX, stepY);
            DrawStarMarker(1, 7, stepX, stepY); DrawStarMarker(7, 7, stepX, stepY);
            for (int i = 0; i < 9; i += 2) { DrawStarMarker(i, 3, stepX, stepY); DrawStarMarker(i, 6, stepX, stepY); }
        }

        private void DrawStarMarker(int col, int row, double stepX, double stepY)
        {
            double centerX = col * stepX + stepX / 2;
            double centerY = row * stepY + stepY / 2;
            double margin = 5;
            var brush = new SolidColorBrush(Color.FromRgb(62, 39, 35));
            if (col > 0) { DrawMarkerCorner(centerX - margin, centerY - margin, -1, -1, brush); }
            if (col < 8) { DrawMarkerCorner(centerX + margin, centerY - margin, 1, -1, brush); }
            if (col > 0) { DrawMarkerCorner(centerX - margin, centerY + margin, -1, 1, brush); }
            if (col < 8) { DrawMarkerCorner(centerX + margin, centerY + margin, 1, 1, brush); }
        }

        private void DrawMarkerCorner(double x, double y, int dirX, int dirY, Brush brush)
        {
            double len = 8;
            DrawLine(x, y, x + dirX * len, y, brush, 1.2);
            DrawLine(x, y, x, y + dirY * len, brush, 1.2);
        }

        private void DrawLine(double x1, double y1, double x2, double y2, Brush brush, double thickness)
        {
            ChessLinesCanvas.Children.Add(new Line { X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, Stroke = brush, StrokeThickness = thickness });
        }

        private void AppendLog(string msg)
        {
            Dispatcher.Invoke(() =>
            {
                LogBox.AppendText($"{DateTime.Now:HH:mm:ss} - {msg}\n");
            });
        }

        private void OnLogBoxTextChanged(object sender, TextChangedEventArgs e)
        {
            if (sender is TextBox tb)
            {
                tb.ScrollToEnd();
            }
        }

        private async Task StartReplayLoopAsync()
        {
            try
            {
                await foreach (var (moves, limit, gameId, result) in _replayChannel.Reader.ReadAllAsync())
                {
                    string gameInfo = gameId > 0 ? $"第 {gameId} 局" : "外部棋谱";
                    AppendLog($"[观战] {gameInfo} 开始播放，共 {moves.Count} 步");
                    
                    // 立即更新 UI 上的局号和结果
                    Dispatcher.Invoke(() => {
                        GameIdLabel.Text = gameId > 0 ? gameId.ToString() : "外部";
                        GameResultLabel.Text = $"({result})";
                    });

                    await ReplayGameInternalAsync(moves, limit, gameId, result);
                }
            }
            catch (ChannelClosedException) { }
        }

        private async Task ReplayGameInternalAsync(List<Move> historyMoves, int maxMovesLimit = 0, int gameId = 0, string result = "")
        {
            Board uiBoard = new Board();
            uiBoard.Reset();

            Dispatcher.Invoke(() => {
                RefreshBoardOnly(uiBoard);
                StepProgressLabel.Text = $"0 / {historyMoves.Count}";
                RemainingStepsLabel.Text = maxMovesLimit > 0 ? maxMovesLimit.ToString() : historyMoves.Count.ToString();
                GameIdLabel.Text = gameId > 0 ? gameId.ToString() : "外部";
                GameResultLabel.Text = string.IsNullOrEmpty(result) ? "" : $"({result})";
            });

            int currentStep = 0;
            int totalGameSteps = historyMoves.Count;
            int effectiveLimit = maxMovesLimit > 0 ? maxMovesLimit : totalGameSteps;

            foreach (var move in historyMoves)
            {
                bool isLastStep = (currentStep == totalGameSteps - 1);
                bool isFastMode = (gameId == -1 && !isLastStep);

                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                });

                if (!isFastMode) await Task.Delay(800);

                uiBoard.Push(move.From, move.To);
                currentStep++;

                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                    _cellButtons[move.To].Tag = "To";
                    StepProgressLabel.Text = $"{currentStep} / {totalGameSteps}";
                    int remaining = effectiveLimit - currentStep;
                    RemainingStepsLabel.Text = Math.Max(0, remaining).ToString();
                });

                if (!isFastMode) await Task.Delay(1500);
            }

            // 每一局最后一步都强制停顿 10 秒
            if (gameId == -1 && historyMoves.Count > 0)
            {
                // 如果是审计失败，对最后一步非法走法进行 3 遍闪烁演示
                var lastMove = historyMoves[^1];
                for (int i = 0; i < 3; i++)
                {
                    Dispatcher.Invoke(() => {
                        _cellButtons[lastMove.From].Tag = "From";
                        _cellButtons[lastMove.To].Tag = "To";
                    });
                    await Task.Delay(500);
                    Dispatcher.Invoke(() => {
                        _cellButtons[lastMove.From].Tag = null;
                        _cellButtons[lastMove.To].Tag = null;
                    });
                    await Task.Delay(500);
                }
            }

            await Task.Delay(10000);

            if (_isManualReplayActive && _replayChannel.Reader.Count == 0)
            {
                _isManualReplayActive = false;
                AppendLog("[观战] 手动棋谱播放完毕，已恢复接收后台最新实战对局...");
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

        public async void OnStartLeagueClick(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_orchestrator.IsTraining) return;
                var config = (TrainingConfig)DataContext;
                if (!int.TryParse(config.PopulationSize, out int populationSize) || populationSize < 2)
                {
                    MessageBox.Show("联赛人口数量必须大于等于 2。", "输入错误", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }
                
                if (populationSize > 100)
                {
                    MessageBox.Show("出于内存限制与并发安全考量，当前版本联赛人口数量不能超过 100。", "输入限制", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                StartLeagueBtn.IsEnabled = false;
                // 使用推荐默认值，AI DNA 将在对局中起主导作用
                await _orchestrator.StartLeagueTrainingAsync(
                    populationSize: populationSize, 
                    maxMoves: 150, 
                    exploreMoves: 40, 
                    materialBias: 0.1f);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"联赛启动异常: {ex.Message}", "致命错误", MessageBoxButton.OK, MessageBoxImage.Error);
                StartLeagueBtn.IsEnabled = true;
            }
        }

        public class TrainingConfig
        {
            public string PopulationSize { get; set; } = "50";
        }

        public void OnReplayLastClick(object sender, RoutedEventArgs e)
        {
            if (_isManualReplayActive)
            {
                _isManualReplayActive = false;
                AppendLog("[系统] 用户手动恢复了后台观战推送。");
                MessageBox.Show("已恢复接收后台最新对局！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        public void OnOpenMasterJsonClick(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "打开 JSON 棋谱",
                Filter = "Chess JSON (*.json)|*.json",
                InitialDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data")
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string json = File.ReadAllText(openFileDialog.FileName);
                    MasterGameData? masterData = null;
                    try { masterData = System.Text.Json.JsonSerializer.Deserialize<MasterGameData>(json); } catch { }

                    if (masterData != null && masterData.MoveHistoryUcci != null && masterData.MoveHistoryUcci.Count > 0)
                    {
                        var moveList = new List<Move>();
                        var tempBoard = new Board(); tempBoard.Reset();
                        var generator = new MoveGenerator();

                        foreach (var rawStr in masterData.MoveHistoryUcci)
                        {
                            string? ucci = NotationConverter.ConvertToUcci(tempBoard, rawStr, generator);
                            if (!string.IsNullOrEmpty(ucci))
                            {
                                var m = NotationConverter.UcciToMove(ucci);
                                if (m != null)
                                {
                                    moveList.Add(m.Value);
                                    tempBoard.Push(m.Value.From, m.Value.To);
                                }
                            }
                        }

                        if (moveList.Count > 0)
                        {
                            _isManualReplayActive = true;
                            AppendLog($"[观战] 加载 JSON 棋谱 ({Path.GetFileName(openFileDialog.FileName)})，共 {moveList.Count} 步。");
                            _replayChannel.Writer.TryWrite((moveList, 0, 0, "手动"));
                        }
                    }
                    else
                    {
                        AppendLog($"[错误] 无法解析该文件为包含对局历史的有效 JSON。当前样本格式不支持通过状态反推走法。");
                    }
                }
                catch (Exception ex) { MessageBox.Show($"加载 JSON 失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error); }
            }
        }

        public async void OnLoadDatasetClick(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_orchestrator.IsTraining) return;
                var openFileDialog = new Microsoft.Win32.OpenFileDialog
                {
                    Title = "选择巨型棋谱数据集",
                    Filter = "支持的数据集 (*.csv;*.pgn;*.txt)|*.csv;*.pgn;*.txt|All files (*.*)|*.*"
                };
                if (openFileDialog.ShowDialog() == true)
                {
                    StartLeagueBtn.IsEnabled = false;
                    AppendLog($"[系统] 准备处理文件: {Path.GetFileName(openFileDialog.FileName)}");
                    await _orchestrator.ProcessDatasetAsync(openFileDialog.FileName);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"导入异常: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                StartLeagueBtn.IsEnabled = true;
            }
        }
    }
}
