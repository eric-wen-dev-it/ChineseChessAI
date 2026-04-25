using ChineseChessAI.Core;
using ChineseChessAI.Training;
using ChineseChessAI.Utils;
using System.IO;
using System.Threading.Channels;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace ChineseChessAI
{
    public partial class MainWindow : Window
    {
        private readonly Button[] _cellButtons = new Button[90];
        private readonly Channel<(List<Move> moves, int limit, int gameId, string result)> _replayChannel;
        private readonly TrainingOrchestrator _orchestrator;
        private readonly bool _autoStartLeague;
        private const string ManualReplayResult = "__manual__";
        private volatile bool _isManualReplayActive;
        private readonly object _replayControlLock = new object();
        private CancellationTokenSource? _activeReplayCts;

        public MainWindow(bool autoStartLeague = false)
        {
            _autoStartLeague = autoStartLeague;
            InitializeComponent();
            InitializeBoardUI();
            Loaded += OnWindowLoaded;

            _replayChannel = Channel.CreateBounded<(List<Move>, int, int, string)>(new BoundedChannelOptions(10)
            {
                FullMode = BoundedChannelFullMode.DropOldest
            });

            _orchestrator = new TrainingOrchestrator();
            _orchestrator.OnLog += AppendLog;
            _orchestrator.OnReplayRequested += (moves, limit, gameId, result) =>
            {
                if (!_isManualReplayActive)
                {
                    _replayChannel.Writer.TryWrite((moves, limit, gameId, result));
                }
            };
            _orchestrator.OnAuditFailureRequested += (history, illegalMove, reason) =>
            {
                if (!_isManualReplayActive)
                {
                    _replayChannel.Writer.TryWrite((history.Concat(new[] { illegalMove }).ToList(), 0, -1, $"Audit failure: {reason}"));
                }
            };
            _orchestrator.OnError += err => Dispatcher.Invoke(() => MessageBox.Show(err, "Error", MessageBoxButton.OK, MessageBoxImage.Error));
            _orchestrator.OnTrainingStopped += () => Dispatcher.Invoke(() => StartLeagueBtn.IsEnabled = true);

            DataContext = new TrainingConfig();
            _ = StartReplayLoopAsync();
        }

        private async void OnWindowLoaded(object? sender, RoutedEventArgs e)
        {
            DrawBoardLines();

            if (!_autoStartLeague)
            {
                return;
            }

            Loaded -= OnWindowLoaded;
            await Dispatcher.InvokeAsync(() => OnStartLeagueClick(this, new RoutedEventArgs()));
        }

        private void InitializeBoardUI()
        {
            ChessBoardGrid.Children.Clear();
            Style pieceStyle = (Style)FindResource("ChessPieceStyle");

            for (int i = 0; i < 90; i++)
            {
                var btn = new Button
                {
                    Style = pieceStyle,
                    Content = string.Empty,
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
            {
                return;
            }

            ChessLinesCanvas.Children.Clear();
            double w = ChessLinesCanvas.ActualWidth;
            double h = ChessLinesCanvas.ActualHeight;
            double stepX = w / 9;
            double stepY = h / 10;

            var gridPen = new SolidColorBrush(Color.FromRgb(62, 39, 35));
            for (int i = 0; i < 10; i++)
            {
                DrawLine(stepX / 2, i * stepY + stepY / 2, w - stepX / 2, i * stepY + stepY / 2, gridPen, 1.5);
            }

            for (int i = 0; i < 9; i++)
            {
                if (i == 0 || i == 8)
                {
                    DrawLine(i * stepX + stepX / 2, stepY / 2, i * stepX + stepX / 2, h - stepY / 2, gridPen, 1.5);
                }
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

            DrawStarMarker(1, 2, stepX, stepY);
            DrawStarMarker(7, 2, stepX, stepY);
            DrawStarMarker(1, 7, stepX, stepY);
            DrawStarMarker(7, 7, stepX, stepY);
            for (int i = 0; i < 9; i += 2)
            {
                DrawStarMarker(i, 3, stepX, stepY);
                DrawStarMarker(i, 6, stepX, stepY);
            }
        }

        private void DrawStarMarker(int col, int row, double stepX, double stepY)
        {
            double centerX = col * stepX + stepX / 2;
            double centerY = row * stepY + stepY / 2;
            double margin = 5;
            var brush = new SolidColorBrush(Color.FromRgb(62, 39, 35));
            if (col > 0)
            {
                DrawMarkerCorner(centerX - margin, centerY - margin, -1, -1, brush);
            }

            if (col < 8)
            {
                DrawMarkerCorner(centerX + margin, centerY - margin, 1, -1, brush);
            }

            if (col > 0)
            {
                DrawMarkerCorner(centerX - margin, centerY + margin, -1, 1, brush);
            }

            if (col < 8)
            {
                DrawMarkerCorner(centerX + margin, centerY + margin, 1, 1, brush);
            }
        }

        private void DrawMarkerCorner(double x, double y, int dirX, int dirY, Brush brush)
        {
            double len = 8;
            DrawLine(x, y, x + dirX * len, y, brush, 1.2);
            DrawLine(x, y, x, y + dirY * len, brush, 1.2);
        }

        private void DrawLine(double x1, double y1, double x2, double y2, Brush brush, double thickness)
        {
            ChessLinesCanvas.Children.Add(new Line
            {
                X1 = x1,
                Y1 = y1,
                X2 = x2,
                Y2 = y2,
                Stroke = brush,
                StrokeThickness = thickness
            });
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
                    if (_isManualReplayActive && (gameId != 0 || result != ManualReplayResult))
                    {
                        continue;
                    }

                    using var replayCts = RegisterActiveReplay();
                    string gameInfo = gameId > 0 ? $"game #{gameId}" : "external replay";
                    AppendLog($"[Replay] Starting {gameInfo}, total {moves.Count} plies.");

                    Dispatcher.Invoke(() =>
                    {
                        GameIdLabel.Text = gameId > 0 ? gameId.ToString() : "external";
                        GameResultLabel.Text = string.IsNullOrEmpty(result) ? string.Empty : $"({result})";
                    });

                    try
                    {
                        await ReplayGameInternalAsync(moves, limit, gameId, result, replayCts.Token);
                    }
                    catch (OperationCanceledException)
                    {
                    }
                    finally
                    {
                        ClearActiveReplay(replayCts);
                    }
                }
            }
            catch (ChannelClosedException)
            {
            }
            catch (Exception ex)
            {
                AppendLog($"[Replay] Worker error: {ex}");
            }
        }

        private CancellationTokenSource RegisterActiveReplay()
        {
            var cts = new CancellationTokenSource();
            lock (_replayControlLock)
            {
                _activeReplayCts = cts;
            }

            return cts;
        }

        private void ClearActiveReplay(CancellationTokenSource completedCts)
        {
            lock (_replayControlLock)
            {
                if (ReferenceEquals(_activeReplayCts, completedCts))
                {
                    _activeReplayCts = null;
                }
            }
        }

        private void CancelActiveReplay()
        {
            CancellationTokenSource? activeReplayCts;
            lock (_replayControlLock)
            {
                activeReplayCts = _activeReplayCts;
            }

            activeReplayCts?.Cancel();
        }

        private void DrainReplayQueue()
        {
            while (_replayChannel.Reader.TryRead(out _))
            {
            }
        }

        private void StartManualReplay(List<Move> moveList, string sourceName)
        {
            _isManualReplayActive = true;
            DrainReplayQueue();
            CancelActiveReplay();

            AppendLog($"[Replay] Loaded {sourceName}, {moveList.Count} plies. Preempting current spectator replay.");
            _replayChannel.Writer.TryWrite((moveList, 0, 0, ManualReplayResult));
        }

        private async Task ReplayGameInternalAsync(
            List<Move> historyMoves,
            int maxMovesLimit = 0,
            int gameId = 0,
            string result = "",
            CancellationToken cancellationToken = default)
        {
            Board uiBoard = new Board();
            uiBoard.Reset();

            Dispatcher.Invoke(() =>
            {
                RefreshBoardOnly(uiBoard);
                StepProgressLabel.Text = $"0 / {historyMoves.Count}";
                RemainingStepsLabel.Text = maxMovesLimit > 0 ? maxMovesLimit.ToString() : historyMoves.Count.ToString();
                GameIdLabel.Text = gameId > 0 ? gameId.ToString() : "external";
                GameResultLabel.Text = string.IsNullOrEmpty(result) ? string.Empty : $"({result})";
            });

            int currentStep = 0;
            int totalGameSteps = historyMoves.Count;
            int effectiveLimit = maxMovesLimit > 0 ? maxMovesLimit : totalGameSteps;

            foreach (var move in historyMoves)
            {
                cancellationToken.ThrowIfCancellationRequested();

                bool isLastStep = currentStep == totalGameSteps - 1;
                bool isFastMode = gameId == -1 && !isLastStep;

                Dispatcher.Invoke(() =>
                {
                    RefreshBoardOnly(uiBoard);
                    _cellButtons[move.From].Tag = "From";
                });

                if (!isFastMode)
                {
                    await Task.Delay(800, cancellationToken);
                }

                cancellationToken.ThrowIfCancellationRequested();
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

                if (!isFastMode)
                {
                    await Task.Delay(1500, cancellationToken);
                }
            }

            if (gameId == -1 && historyMoves.Count > 0)
            {
                var lastMove = historyMoves[^1];
                for (int i = 0; i < 3; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Dispatcher.Invoke(() =>
                    {
                        _cellButtons[lastMove.From].Tag = "From";
                        _cellButtons[lastMove.To].Tag = "To";
                    });
                    await Task.Delay(500, cancellationToken);
                    Dispatcher.Invoke(() =>
                    {
                        _cellButtons[lastMove.From].Tag = null;
                        _cellButtons[lastMove.To].Tag = null;
                    });
                    await Task.Delay(500, cancellationToken);
                }
            }

            await Task.Delay(10000, cancellationToken);

            if (_isManualReplayActive && _replayChannel.Reader.Count == 0)
            {
                _isManualReplayActive = false;
                AppendLog("[Replay] Manual replay finished. Auto spectator feed restored.");
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
                if (_orchestrator.IsTraining)
                {
                    return;
                }

                var config = (TrainingConfig)DataContext;
                if (!int.TryParse(config.PopulationSize, out int populationSize) || populationSize < 2)
                {
                    MessageBox.Show("Population must be at least 2.", "Input Error", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                if (populationSize > 100)
                {
                    MessageBox.Show("Population cannot exceed 100 in the current build.", "Input Limit", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }

                StartLeagueBtn.IsEnabled = false;
                await _orchestrator.StartLeagueTrainingAsync(
                    populationSize: populationSize,
                    maxMoves: 150,
                    exploreMoves: 40,
                    materialBias: 0.1f);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"League start failed: {ex.Message}", "Fatal Error", MessageBoxButton.OK, MessageBoxImage.Error);
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
                DrainReplayQueue();
                CancelActiveReplay();
                AppendLog("[Replay] Manual replay canceled. Auto spectator feed resumed.");
                MessageBox.Show("Auto spectator feed resumed.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        public void OnOpenMasterJsonClick(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Title = "Open JSON Replay",
                Filter = "Chess JSON (*.json)|*.json",
                InitialDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data")
            };

            if (openFileDialog.ShowDialog() != true)
            {
                return;
            }

            try
            {
                string json = File.ReadAllText(openFileDialog.FileName);
                MasterGameData? masterData = null;
                try
                {
                    masterData = System.Text.Json.JsonSerializer.Deserialize<MasterGameData>(json);
                }
                catch
                {
                }

                if (masterData == null || masterData.MoveHistoryUcci == null || masterData.MoveHistoryUcci.Count == 0)
                {
                    AppendLog("[Replay] The selected JSON does not contain a playable MoveHistoryUcci.");
                    return;
                }

                var moveList = new List<Move>();
                var tempBoard = new Board();
                tempBoard.Reset();
                var generator = new MoveGenerator();

                foreach (var rawStr in masterData.MoveHistoryUcci)
                {
                    string? ucci = NotationConverter.ConvertToUcci(tempBoard, rawStr, generator);
                    if (string.IsNullOrEmpty(ucci))
                    {
                        continue;
                    }

                    Move? move = NotationConverter.UcciToMove(ucci);
                    if (move == null)
                    {
                        continue;
                    }

                    moveList.Add(move.Value);
                    tempBoard.Push(move.Value.From, move.Value.To);
                }

                if (moveList.Count == 0)
                {
                    AppendLog("[Replay] The selected JSON contains move history, but no playable moves were produced.");
                    return;
                }

                StartManualReplay(moveList, Path.GetFileName(openFileDialog.FileName));
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to load JSON: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public async void OnLoadDatasetClick(object sender, RoutedEventArgs e)
        {
            try
            {
                if (_orchestrator.IsTraining)
                {
                    return;
                }

                var openFileDialog = new Microsoft.Win32.OpenFileDialog
                {
                    Title = "Select Dataset",
                    Filter = "Supported datasets (*.csv;*.pgn;*.txt)|*.csv;*.pgn;*.txt|All files (*.*)|*.*"
                };

                if (openFileDialog.ShowDialog() == true)
                {
                    StartLeagueBtn.IsEnabled = false;
                    AppendLog($"[System] Loading dataset: {Path.GetFileName(openFileDialog.FileName)}");
                    await _orchestrator.ProcessDatasetAsync(openFileDialog.FileName);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Dataset import failed: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                StartLeagueBtn.IsEnabled = true;
            }
        }
    }
}
