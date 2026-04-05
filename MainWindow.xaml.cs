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
        private Channel<(List<Move> moves, int limit)> _replayChannel;
        private TrainingOrchestrator _orchestrator;
        private volatile bool _isManualReplayActive = false;

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoardUI();
            this.Loaded += (s, e) => DrawBoardLines();

            _replayChannel = Channel.CreateBounded<(List<Move>, int)>(new BoundedChannelOptions(1)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });

            _orchestrator = new TrainingOrchestrator();
            _orchestrator.OnLog += msg => AppendLog(msg);
            _orchestrator.OnLossUpdated += loss => Dispatcher.Invoke(() => LossLabel.Text = loss.ToString("F4"));

            _orchestrator.OnReplayRequested += (moves, limit) =>
            {
                if (!_isManualReplayActive)
                {
                    _replayChannel.Writer.TryWrite((moves, limit));
                }
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
                LogBox.ScrollToEnd();
            });
        }

        private async Task StartReplayLoopAsync()
        {
            try
            {
                await foreach (var (moves, limit) in _replayChannel.Reader.ReadAllAsync())
                {
                    AppendLog($"[观战] 开始播放对局动画，步数: {moves.Count}");
                    await ReplayGameInternalAsync(moves, limit);
                }
            }
            catch (ChannelClosedException) { }
        }

        private async Task ReplayGameInternalAsync(List<Move> historyMoves, int maxMovesLimit = 0)
        {
            Board uiBoard = new Board();
            uiBoard.Reset();

            Dispatcher.Invoke(() => {
                RefreshBoardOnly(uiBoard);
                StepProgressLabel.Text = $"0 / {historyMoves.Count}";
                RemainingStepsLabel.Text = maxMovesLimit > 0 ? maxMovesLimit.ToString() : historyMoves.Count.ToString();
            });

            int currentStep = 0;
            int totalGameSteps = historyMoves.Count;
            int effectiveLimit = maxMovesLimit > 0 ? maxMovesLimit : totalGameSteps;

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

                await Task.Delay(800);

                if (_replayChannel.Reader.Count > 0) break;

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

                await Task.Delay(1500);
            }

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

        public async void OnStartLeagueClick(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_orchestrator.IsTraining) return;
                var config = (TrainingConfig)DataContext;
                if (!int.TryParse(config.MaxMoves, out int maxMoves) || maxMoves <= 0) return;
                if (!int.TryParse(config.ExploreMoves, out int exploreMoves) || exploreMoves < 0) return;
                if (!float.TryParse(config.MaterialBias, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float materialBias)) return;
                if (!int.TryParse(config.PopulationSize, out int populationSize) || populationSize <= 0) return;

                StartLeagueBtn.IsEnabled = false;
                await _orchestrator.StartLeagueTrainingAsync(populationSize, maxMoves, exploreMoves, materialBias);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"联赛启动异常: {ex.Message}", "致命错误", MessageBoxButton.OK, MessageBoxImage.Error);
                StartLeagueBtn.IsEnabled = true;
            }
        }

        public class TrainingConfig
        {
            public string MaxMoves { get; set; } = "150";
            public string ExploreMoves { get; set; } = "40";
            public string MaterialBias { get; set; } = "0.6";
            public string PopulationSize { get; set; } = "10000";
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

        public void OnLoadFileClick(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "选择文本棋谱文件",
                Filter = "Text/PGN files (*.txt;*.pgn)|*.txt;*.pgn|All files (*.*)|*.*",
                InitialDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "game_logs")
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string fileContent = File.ReadAllText(openFileDialog.FileName);
                    string movesStr = "";
                    if (fileContent.Contains("棋谱:"))
                    {
                        var lines = fileContent.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
                        foreach (var line in lines) if (line.StartsWith("棋谱:")) { movesStr = line.Substring(3).Trim(); break; }
                    }
                    else
                    {
                        movesStr = System.Text.RegularExpressions.Regex.Replace(fileContent, @"\[[^\]]*\]", "");
                        movesStr = System.Text.RegularExpressions.Regex.Replace(movesStr, @"\{[^}]*\}", "");
                        movesStr = System.Text.RegularExpressions.Regex.Replace(movesStr, @"\b\d+\.", "");
                        movesStr = movesStr.Replace("1-0", "").Replace("0-1", "").Replace("1/2-1/2", "").Replace("*", "");
                    }

                    var moveList = new List<Move>();
                    var rawMoves = movesStr.Split(new[] { ' ', '\n', '\r', '\t', ',' }, StringSplitOptions.RemoveEmptyEntries);
                    var tempBoard = new Board(); tempBoard.Reset();
                    var generator = new MoveGenerator();
                    foreach (var mStr in rawMoves)
                    {
                        string? ucci = NotationConverter.ConvertToUcci(tempBoard, mStr, generator);
                        if (!string.IsNullOrEmpty(ucci))
                        {
                            var move = NotationConverter.UcciToMove(ucci);
                            if (move != null) { moveList.Add(move.Value); tempBoard.Push(move.Value.From, move.Value.To); }
                        }
                    }
                    if (moveList.Count > 0)
                    {
                        _isManualReplayActive = true;
                        AppendLog($"[观战] 已成功加载文本棋谱，共 {moveList.Count} 步。");
                        _replayChannel.Writer.TryWrite((moveList, 0));
                    }
                }
                catch (Exception ex) { MessageBox.Show($"解析失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error); }
            }
        }

        public void OnOpenMasterJsonClick(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "打开大师 JSON 棋谱",
                Filter = "Master JSON (*.json)|*.json",
                InitialDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "master_data")
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    string json = File.ReadAllText(openFileDialog.FileName);
                    var masterData = System.Text.Json.JsonSerializer.Deserialize<MasterGameData>(json);

                    if (masterData != null && masterData.MoveHistoryUcci != null && masterData.MoveHistoryUcci.Count > 0)
                    {
                        var moveList = new List<Move>();
                        var tempBoard = new Board(); tempBoard.Reset();
                        var generator = new MoveGenerator();

                        // 【核心改进】：维持棋盘状态逐步解析，确保兼容代数记谱法 (如 C2.5)
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
                            AppendLog($"[观战] 精准加载 JSON 棋谱 ({Path.GetFileName(openFileDialog.FileName)})，共 {moveList.Count} 步。");
                            _replayChannel.Writer.TryWrite((moveList, 0));
                        }
                    }
                    else
                    {
                        // 兼容尝试：旧版 JSON 处理
                        var examples = System.Text.Json.JsonSerializer.Deserialize<List<TrainingExample>>(json);
                        if (examples != null && examples.Count > 1)
                        {
                            var moveList = new List<Move>();
                            for (int i = 0; i < examples.Count - 1; i++)
                            {
                                var m = ReconstructMove(examples[i].State, examples[i + 1].State);
                                if (m != null) moveList.Add(m.Value);
                            }
                            _isManualReplayActive = true;
                            AppendLog($"[观战] 从旧版 JSON 还原棋谱，共 {moveList.Count} 步。");
                            _replayChannel.Writer.TryWrite((moveList, 0));
                        }
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

        private Move? ReconstructMove(float[] stateBefore, float[] stateAfter)
        {
            var piecesBefore = GetPieceMap(stateBefore);
            var piecesAfter = GetPieceMap(stateAfter);
            int from = -1, to = -1;
            foreach (var pos in piecesBefore.Keys) if (!piecesAfter.ContainsKey(pos)) from = pos;
            foreach (var pos in piecesAfter.Keys) if (!piecesBefore.ContainsKey(pos) || piecesBefore[pos] != piecesAfter[pos]) to = pos;
            if (from != -1 && to != -1) return new Move(from, to);
            return null;
        }

        private Dictionary<int, int> GetPieceMap(float[] state)
        {
            var map = new Dictionary<int, int>();
            for (int layer = 0; layer < 14; layer++)
                for (int pos = 0; pos < 90; pos++)
                    if (state[layer * 90 + pos] > 0.5f) map[pos] = layer;
            return map;
        }
    }
}
