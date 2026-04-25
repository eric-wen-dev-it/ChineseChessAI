using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Traditional;
using Microsoft.Win32;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace ChineseChessAI.Play
{
    public enum PlayPlayerKind
    {
        Human,
        Mcts,
        Traditional,
        Pikafish
    }

    public partial class MainWindow : Window
    {
        private const int DefaultMaxMoves = 150;

        private readonly Button[] _cellButtons = new Button[90];
        private readonly GameRuleSession _session = new();
        private readonly ChineseChessRuleEngine _rules = new();
        private readonly HashSet<int> _legalTargets = new();
        private readonly PlayStrengthSettings _playSettings = LoadPlaySettings();

        private CChessNet? _model;
        private MCTSEngine? _mctsEngine;
        private TraditionalEngine? _traditionalEngine;
        private PikafishEngineClient? _pikafishEngine;
        private OpeningBook? _openingBook;
        private CancellationTokenSource? _aiCts;

        private int? _selectedIndex;
        private string _loadedModelPath = string.Empty;
        private bool _isAiThinking;
        private bool _gameOver;

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoard();
            Loaded += (_, _) => DrawBoardLines();

            SimulationsTextBox.Text = _playSettings.DefaultSimulations.ToString();
            ModelPathTextBox.Text = FindDefaultEnginePath();
            RefreshEngineTypeUi();
            RefreshBoard();
            UpdateUiState("Choose a model and start a game.");

            Closed += (_, _) => DisposeResources();
        }

        private void InitializeBoard()
        {
            BoardGrid.Children.Clear();
            Style pieceStyle = (Style)FindResource("ChessPieceStyle");

            for (int i = 0; i < _cellButtons.Length; i++)
            {
                var button = new Button
                {
                    Style = pieceStyle,
                    DataContext = i,
                    FontSize = 26,
                    FontFamily = new FontFamily("KaiTi"),
                    FontWeight = FontWeights.Bold,
                    Margin = new Thickness(2),
                    Foreground = Brushes.Black,
                    FocusVisualStyle = null,
                    Tag = null
                };
                button.Click += OnBoardCellClick;
                _cellButtons[i] = button;
                BoardGrid.Children.Add(button);
            }
        }

        private void DrawBoardLines()
        {
            if (ChessLinesCanvas == null)
                return;

            ChessLinesCanvas.Children.Clear();
            double w = ChessLinesCanvas.ActualWidth;
            double h = ChessLinesCanvas.ActualHeight;
            if (w <= 0 || h <= 0)
                return;

            double stepX = w / 9;
            double stepY = h / 10;
            var gridPen = new SolidColorBrush(Color.FromRgb(62, 39, 35));

            for (int i = 0; i < 10; i++)
                DrawLine(stepX / 2, i * stepY + stepY / 2, w - stepX / 2, i * stepY + stepY / 2, gridPen, 1.5);

            for (int i = 0; i < 9; i++)
            {
                double x = i * stepX + stepX / 2;
                if (i == 0 || i == 8)
                {
                    DrawLine(x, stepY / 2, x, h - stepY / 2, gridPen, 1.5);
                    continue;
                }

                DrawLine(x, stepY / 2, x, 4 * stepY + stepY / 2, gridPen, 1.5);
                DrawLine(x, 5 * stepY + stepY / 2, x, h - stepY / 2, gridPen, 1.5);
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
                DrawMarkerCorner(centerX - margin, centerY - margin, -1, -1, brush);
            if (col < 8)
                DrawMarkerCorner(centerX + margin, centerY - margin, 1, -1, brush);
            if (col > 0)
                DrawMarkerCorner(centerX - margin, centerY + margin, -1, 1, brush);
            if (col < 8)
                DrawMarkerCorner(centerX + margin, centerY + margin, 1, 1, brush);
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

        private async void OnNewGameClick(object sender, RoutedEventArgs e)
        {
            if (_isAiThinking)
                return;

            if (!await TryCreateEnginesAsync())
                return;

            _session.Reset();
            if (_pikafishEngine != null)
                await _pikafishEngine.NewGameAsync(CancellationToken.None);
            _gameOver = false;
            ResetSelection();
            LogBox.Clear();

            AppendLog($"New game. Red={GetPlayerKind(true)}, Black={GetPlayerKind(false)}.");
            RefreshBoard();
            UpdateUiState("Game started.");

            await ContinueAutoPlayAsync();
        }

        private async void OnAiMoveClick(object sender, RoutedEventArgs e)
        {
            if (_gameOver || _isAiThinking)
                return;

            if (!await TryCreateEnginesAsync())
                return;

            if (IsHumanTurn())
            {
                UpdateUiState("It is currently your turn.");
                return;
            }

            await ContinueAutoPlayAsync();
        }

        private async void OnTraditionalDemoClick(object sender, RoutedEventArgs e)
        {
            if (_isAiThinking)
                return;

            DisposeEngine();
            _session.Reset();
            _gameOver = false;
            ResetSelection();
            LogBox.Clear();
            RefreshBoard();
            AppendLog("Traditional demo started: Red Traditional vs Black Traditional.");

            var token = ResetAiToken();
            _isAiThinking = true;
            UpdateUiState("Traditional demo running...");
            RefreshBoard();

            try
            {
                var book = GetOrLoadOpeningBook();
                var bookMode = book.PositionCount > 0 ? OpeningBookMode.Weighted : OpeningBookMode.Off;
                var redEngine = new TraditionalEngine(new TraditionalEngineOptions
                {
                    OpeningBook = book,
                    OpeningBookMode = bookMode,
                    MoveOrderingBook = OpeningBook.LoadDefaultCache(maxPly: 80, fileName: "master_move_ordering.json"),
                    RootParallelism = ResolveTraditionalRootParallelism()
                });
                var blackEngine = new TraditionalEngine(new TraditionalEngineOptions
                {
                    OpeningBook = book,
                    OpeningBookMode = bookMode,
                    MoveOrderingBook = OpeningBook.LoadDefaultCache(maxPly: 80, fileName: "master_move_ordering.json"),
                    RootParallelism = ResolveTraditionalRootParallelism()
                });
                const int demoMaxPly = 80;
                const int demoMoveTimeMs = 5000;
                int depth = 4;
                int quietPly = 0;
                var positionCounts = new Dictionary<ulong, int>
                {
                    [_session.Board.CurrentHash] = 1
                };
                AppendLog($"Traditional demo: opening_book={bookMode}, depth={depth}, move_time={demoMoveTimeMs}ms, max_ply={demoMaxPly}.");

                for (int ply = 0; ply < demoMaxPly && !_gameOver; ply++)
                {
                    token.ThrowIfCancellationRequested();

                    var activeEngine = _session.Board.IsRedTurn ? redEngine : blackEngine;
                    string actor = _session.Board.IsRedTurn ? "Red Traditional" : "Black Traditional";
                    UpdateUiState($"{actor} thinking at depth {depth}...");

                    var result = await Task.Run(
                        () => activeEngine.Search(_session.Board.Clone(), new SearchLimits(depth, demoMoveTimeMs, 6), token),
                        token);

                    if (result.BestMove.From == result.BestMove.To)
                    {
                        AppendLog($"{actor}: no legal move.");
                        TryFinishGame();
                        break;
                    }

                    AppendLog($"{actor} search: depth={result.Depth}, score={result.Score}, nodes={result.Nodes}, time={result.Elapsed.TotalMilliseconds:F0}ms.");
                    bool irreversible = _session.Board.GetPiece(result.BestMove.To) != 0 || Math.Abs(_session.Board.GetPiece(result.BestMove.From)) == 7;
                    ApplyMove(result.BestMove, actor);
                    RefreshBoard();

                    if (TryFinishGame())
                        break;

                    if (irreversible)
                    {
                        quietPly = 0;
                        positionCounts.Clear();
                    }
                    else
                    {
                        quietPly++;
                    }

                    positionCounts.TryGetValue(_session.Board.CurrentHash, out int count);
                    positionCounts[_session.Board.CurrentHash] = count + 1;
                    if (count + 1 >= 3)
                    {
                        _gameOver = true;
                        UpdateUiState("Traditional demo ended by repetition.");
                        AppendLog("Traditional demo ended by threefold repetition.");
                        break;
                    }

                    if (quietPly >= 80)
                    {
                        _gameOver = true;
                        UpdateUiState("Traditional demo ended by no-progress rule.");
                        AppendLog("Traditional demo ended by no-progress rule.");
                        break;
                    }

                    await Task.Delay(450, token);
                }

                if (!_gameOver)
                {
                    _gameOver = true;
                    UpdateUiState("Traditional demo ended by move limit.");
                    AppendLog("Traditional demo ended by move limit.");
                }
            }
            catch (OperationCanceledException)
            {
                UpdateUiState("Traditional demo canceled.");
                AppendLog("Traditional demo canceled.");
            }
            catch (Exception ex)
            {
                UpdateUiState($"Traditional demo failed: {ex.Message}");
                MessageBox.Show(ex.Message, "Traditional demo failed", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                _isAiThinking = false;
                RefreshBoard();
                UpdateUiState(_gameOver ? StatusTextBlock.Text : "Traditional demo ready.");
            }
        }

        private void OnResignClick(object sender, RoutedEventArgs e)
        {
            if (_gameOver || _isAiThinking)
                return;

            var confirm = MessageBox.Show(
                "确认认输并结束当前对局？",
                "确认认输",
                MessageBoxButton.YesNo,
                MessageBoxImage.Question);

            if (confirm != MessageBoxResult.Yes)
                return;

            _gameOver = true;
            ResetSelection();

            string winner = _session.Board.IsRedTurn ? "Black" : "Red";
            UpdateUiState($"{winner} wins by resignation.");
            AppendLog($"Game over: {winner} wins by resignation.");
            RefreshBoard();
            ShowGameOverDialog(winner, "resignation");
        }

        private void OnBrowseModelClick(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = NeedsPikafish() ? "Select Pikafish executable" : "Select model file",
                Filter = NeedsPikafish()
                    ? "Executable (*.exe)|*.exe|All files (*.*)|*.*"
                    : "Torch model (*.pt)|*.pt|All files (*.*)|*.*"
            };

            if (dialog.ShowDialog() == true)
                ModelPathTextBox.Text = dialog.FileName;
        }

        private void OnPlayerTypeChanged(object sender, SelectionChangedEventArgs e)
        {
            RefreshEngineTypeUi();
            DisposeEngine();
        }

        private async void OnBoardCellClick(object sender, RoutedEventArgs e)
        {
            if (_gameOver || _isAiThinking || !IsHumanTurn())
                return;

            if (sender is not Button button || button.DataContext is not int index)
                return;

            sbyte piece = _session.Board.GetPiece(index);
            bool isOwnPiece = piece != 0 && ((piece > 0) == _session.Board.IsRedTurn);

            if (_selectedIndex is null)
            {
                if (isOwnPiece)
                {
                    SelectSquare(index);
                    RefreshBoard();
                }
                return;
            }

            if (_selectedIndex.Value == index)
            {
                ResetSelection();
                RefreshBoard();
                return;
            }

            if (_legalTargets.Contains(index))
            {
                ApplyMove(new Move(_selectedIndex.Value, index), "Human");
                ResetSelection();
                RefreshBoard();

                if (!TryFinishGame())
                    await ContinueAutoPlayAsync();
                return;
            }

            if (isOwnPiece)
            {
                SelectSquare(index);
                RefreshBoard();
                return;
            }

            ResetSelection();
            RefreshBoard();
        }

        private void SelectSquare(int index)
        {
            _selectedIndex = index;
            _legalTargets.Clear();

            foreach (var move in _rules.GetLegalMoves(_session.Board).Where(m => m.From == index))
                _legalTargets.Add(move.To);

            UpdateUiState(_legalTargets.Count > 0 ? $"Selected {Board.GetPieceName(_session.Board.GetPiece(index))}." : "That piece has no legal moves.");
        }

        private async Task ContinueAutoPlayAsync()
        {
            while (!_gameOver && !IsHumanTurn())
            {
                await MakeAiMoveAsync();
                if (_gameOver || IsHumanTurn())
                    break;

                await Task.Delay(200);
            }
        }

        private async Task MakeAiMoveAsync()
        {
            var playerKind = GetCurrentPlayerKind();
            if (playerKind == PlayPlayerKind.Human)
                return;

            if ((playerKind == PlayPlayerKind.Mcts && _mctsEngine == null)
                || (playerKind == PlayPlayerKind.Traditional && _traditionalEngine == null)
                || (playerKind == PlayPlayerKind.Pikafish && _pikafishEngine == null))
                return;

            int searchBudget = ParseSearchBudget(playerKind);
            var token = ResetAiToken();

            _isAiThinking = true;
            ResetSelection();
            RefreshBoard();
            string side = _session.Board.IsRedTurn ? "Red" : "Black";
            UpdateUiState(playerKind switch
            {
                PlayPlayerKind.Traditional => $"{side} Traditional thinking at depth {searchBudget}, {_playSettings.TraditionalMoveTimeMs}ms cap...",
                PlayPlayerKind.Pikafish => $"{side} Pikafish thinking at depth {searchBudget}, {_playSettings.PikafishMoveTimeMs}ms cap...",
                _ => $"{side} MCTS thinking with {searchBudget} simulations..."
            });

            try
            {
                Move move;
                if (playerKind == PlayPlayerKind.Traditional)
                {
                    var engine = _traditionalEngine ?? throw new InvalidOperationException("Traditional engine is not initialized.");
                    var boardSnapshot = _session.Board.Clone();
                    var limits = new SearchLimits(searchBudget, _playSettings.TraditionalMoveTimeMs, 4);
                    var result = await Task.Run(
                        () => engine.Search(boardSnapshot, limits, token),
                        token);
                    move = result.BestMove;
                    AppendLog($"{side} Traditional search: depth={result.Depth}, score={result.Score}, nodes={result.Nodes}, time={result.Elapsed.TotalMilliseconds:F0}ms, complete={result.Completed}.");
                }
                else if (playerKind == PlayPlayerKind.Pikafish)
                {
                    var engine = _pikafishEngine ?? throw new InvalidOperationException("Pikafish engine is not initialized.");
                    string bestMove = await engine.GetBestMoveAsync(_session.UcciHistory, searchBudget, _playSettings.PikafishMoveTimeMs, token);
                    if (!_session.TryResolveUcci(bestMove, out move, out string reason))
                        throw new InvalidOperationException($"Pikafish returned illegal move {bestMove}: {reason}");

                    AppendLog($"{side} Pikafish bestmove: {bestMove}.");
                }
                else
                {
                    var engine = _mctsEngine ?? throw new InvalidOperationException("MCTS engine is not initialized.");
                    (move, _) = await engine.GetMoveWithProbabilitiesAsArrayAsync(
                        _session.Board,
                        searchBudget,
                        _session.MoveHistory.Count,
                        DefaultMaxMoves,
                        token,
                        addRootNoise: _playSettings.AddRootNoise);
                }

                ApplyMove(move, side);
                RefreshBoard();
                TryFinishGame();
            }
            catch (OperationCanceledException)
            {
                UpdateUiState("AI search canceled.");
            }
            catch (Exception ex)
            {
                UpdateUiState($"AI move failed: {ex.Message}");
                MessageBox.Show(ex.Message, "AI move failed", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                _isAiThinking = false;
                RefreshBoard();
                UpdateUiState(_gameOver ? StatusTextBlock.Text : (IsHumanTurn() ? "Human move." : "AI ready."));
            }
        }

        private void ApplyMove(Move move, string actor)
        {
            string moveName = _session.Board.GetChineseMoveName(move.From, move.To);
            _session.ApplyMove(move);
            AppendLog($"{actor}: {moveName} ({move})");
        }

        private bool TryFinishGame()
        {
            if (!HasKing(1))
            {
                _gameOver = true;
                const string winner = "Black";
                UpdateUiState($"{winner} wins.");
                AppendLog($"Game over: {winner} wins.");
                ShowGameOverDialog(winner, null);
                return true;
            }

            if (!HasKing(-1))
            {
                _gameOver = true;
                const string winner = "Red";
                UpdateUiState($"{winner} wins.");
                AppendLog($"Game over: {winner} wins.");
                ShowGameOverDialog(winner, null);
                return true;
            }

            var legalMoves = _rules.GetLegalMoves(_session.Board);
            if (legalMoves.Count == 0)
            {
                _gameOver = true;
                bool currentSideSafe = _rules.IsKingSafe(_session.Board, _session.Board.IsRedTurn);
                string winner = _session.Board.IsRedTurn ? "Black" : "Red";
                string ending = currentSideSafe ? "stalemate" : "checkmate";
                UpdateUiState($"{winner} wins by {ending}.");
                AppendLog($"Game over: {winner} wins by {ending}.");
                ShowGameOverDialog(winner, ending);
                return true;
            }

            return false;
        }

        private void ShowGameOverDialog(string winner, string? ending)
        {
            bool humanWon = (winner == "Red" && GetPlayerKind(true) == PlayPlayerKind.Human)
                || (winner == "Black" && GetPlayerKind(false) == PlayPlayerKind.Human);
            string resultLine = humanWon ? "你赢了。" : "你输了。";
            string winnerLine = winner == "Red" ? "红方获胜。" : "黑方获胜。";
            string endingLine = ending switch
            {
                "checkmate" => "终局类型：将死。",
                "stalemate" => "终局类型：困毙。",
                "resignation" => "终局类型：认输。",
                _ => "终局类型：将帅被吃。"
            };

            MessageBox.Show(
                $"{resultLine}\n{winnerLine}\n{endingLine}",
                "对局结束",
                MessageBoxButton.OK,
                humanWon ? MessageBoxImage.Information : MessageBoxImage.Warning);
        }

        private async Task<bool> TryCreateEnginesAsync()
        {
            bool needsTraditional = GetPlayerKind(true) == PlayPlayerKind.Traditional || GetPlayerKind(false) == PlayPlayerKind.Traditional;
            bool needsMcts = GetPlayerKind(true) == PlayPlayerKind.Mcts || GetPlayerKind(false) == PlayPlayerKind.Mcts;
            bool needsPikafish = GetPlayerKind(true) == PlayPlayerKind.Pikafish || GetPlayerKind(false) == PlayPlayerKind.Pikafish;

            if (!needsTraditional)
                _traditionalEngine = null;
            if (!needsPikafish)
            {
                _pikafishEngine?.Dispose();
                _pikafishEngine = null;
            }
            if (!needsMcts)
            {
                _mctsEngine?.Dispose();
                _mctsEngine = null;
                _model?.Dispose();
                _model = null;
            }

            if (needsTraditional && _traditionalEngine == null)
            {
                var book = GetOrLoadOpeningBook();
                _traditionalEngine = new TraditionalEngine(new TraditionalEngineOptions
                {
                    OpeningBook = book,
                    OpeningBookMode = book.PositionCount > 0 ? OpeningBookMode.Weighted : OpeningBookMode.Off,
                    MoveOrderingBook = OpeningBook.LoadDefaultCache(maxPly: 80, fileName: "master_move_ordering.json"),
                    RootParallelism = ResolveTraditionalRootParallelism()
                });
                _loadedModelPath = string.Empty;
                AppendLog(book.PositionCount > 0
                    ? $"Traditional engine ready. Opening book positions: {book.PositionCount}."
                    : "Traditional engine ready. Opening book not found.");
                AppendLog($"Traditional settings: depth={_playSettings.TraditionalDepth}, move_time={_playSettings.TraditionalMoveTimeMs}ms, root_parallelism={ResolveTraditionalRootParallelism()}.");
            }

            if (needsPikafish)
            {
                string pikafishPath = ModelPathTextBox.Text.Trim();
                if (string.IsNullOrWhiteSpace(pikafishPath))
                {
                    UpdateUiState("Select Pikafish executable first.");
                    return false;
                }

                string pikafishFullPath = Path.GetFullPath(pikafishPath);
                if (!File.Exists(pikafishFullPath))
                {
                    UpdateUiState("Pikafish executable does not exist.");
                    MessageBox.Show(pikafishFullPath, "Pikafish executable not found", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return false;
                }

                if (_pikafishEngine != null && string.Equals(_loadedModelPath, pikafishFullPath, StringComparison.OrdinalIgnoreCase))
                {
                    // Already loaded.
                }
                else
                {
                    _pikafishEngine?.Dispose();
                    _pikafishEngine = null;
                    try
                    {
                        _pikafishEngine = new PikafishEngineClient(pikafishFullPath);
                        await _pikafishEngine.InitializeAsync(CancellationToken.None);
                        await _pikafishEngine.NewGameAsync(CancellationToken.None);
                        _loadedModelPath = pikafishFullPath;
                        AppendLog($"Loaded Pikafish: {Path.GetFileName(pikafishFullPath)}");
                        AppendLog($"Pikafish settings: depth={_playSettings.TraditionalDepth}, move_time={_playSettings.PikafishMoveTimeMs}ms.");
                    }
                    catch (Exception ex)
                    {
                        _pikafishEngine?.Dispose();
                        _pikafishEngine = null;
                        UpdateUiState($"Pikafish load failed: {ex.Message}");
                        MessageBox.Show(ex.Message, "Pikafish load failed", MessageBoxButton.OK, MessageBoxImage.Error);
                        return false;
                    }
                }
            }

            if (!needsMcts)
                return true;

            string rawPath = ModelPathTextBox.Text.Trim();
            if (string.IsNullOrWhiteSpace(rawPath))
            {
                UpdateUiState("Select a model file first.");
                return false;
            }

            string fullPath = Path.GetFullPath(rawPath);
            if (!File.Exists(fullPath))
            {
                UpdateUiState("Model file does not exist.");
                MessageBox.Show(fullPath, "Model file not found", MessageBoxButton.OK, MessageBoxImage.Warning);
                return false;
            }

            if (_mctsEngine != null && string.Equals(_loadedModelPath, fullPath, StringComparison.OrdinalIgnoreCase))
                return true;

            _mctsEngine?.Dispose();
            _mctsEngine = null;
            _model?.Dispose();
            _model = null;

            try
            {
                var model = new CChessNet();
                ModelManager.LoadModel(model, fullPath);

                _model = model;
                _mctsEngine = new MCTSEngine(model, _playSettings.BatchSize, _playSettings.CPuct);
                _loadedModelPath = fullPath;

                AppendLog($"Loaded model: {Path.GetFileName(fullPath)}");
                AppendLog($"Play settings: sims={_playSettings.DefaultSimulations}, batch={_playSettings.BatchSize}, c_puct={_playSettings.CPuct:F2}, root_noise={_playSettings.AddRootNoise}.");
                return true;
            }
            catch (Exception ex)
            {
                DisposeEngine();
                UpdateUiState($"Model load failed: {ex.Message}");
                MessageBox.Show(ex.Message, "Model load failed", MessageBoxButton.OK, MessageBoxImage.Error);
                return false;
            }
        }

        private void RefreshBoard()
        {
            for (int boardIndex = 0; boardIndex < _cellButtons.Length; boardIndex++)
            {
                var button = _cellButtons[GetDisplayIndex(boardIndex)];
                sbyte piece = _session.Board.GetPiece(boardIndex);

                button.DataContext = boardIndex;
                button.Content = piece == 0 ? string.Empty : Board.GetPieceName(piece);
                button.Foreground = piece > 0 ? Brushes.Firebrick : Brushes.Black;
                button.Tag = GetCellHighlightTag(boardIndex);
            }

            MoveHistoryTextBox.Text = _session.Board.GetMoveHistoryString();
            BoardGrid.IsEnabled = !_gameOver && !_isAiThinking && IsHumanTurn();
        }

        private int GetDisplayIndex(int boardIndex)
        {
            return boardIndex;
        }

        private string? GetCellHighlightTag(int index)
        {
            if (_selectedIndex == index)
                return "Selected";

            if (_legalTargets.Contains(index))
                return "Legal";

            if (_session.Board.LastMove is Move lastMove)
            {
                if (lastMove.From == index)
                    return "From";
                if (lastMove.To == index)
                    return "To";
            }

            return null;
        }

        private void UpdateUiState(string status)
        {
            StatusTextBlock.Text = status;
            TurnTextBlock.Text = _gameOver
                ? "Turn: game over"
                : $"Turn: {(_session.Board.IsRedTurn ? "Red" : "Black")}";
            SideTextBlock.Text = $"Red: {GetPlayerKind(true)} | Black: {GetPlayerKind(false)}";

            NewGameButton.IsEnabled = !_isAiThinking;
            AiMoveButton.IsEnabled = !_gameOver && !_isAiThinking;
            TraditionalDemoButton.IsEnabled = !_isAiThinking;
            ResignButton.IsEnabled = !_gameOver && !_isAiThinking;
        }

        private void ResetSelection()
        {
            _selectedIndex = null;
            _legalTargets.Clear();
        }

        private bool IsHumanTurn() => GetCurrentPlayerKind() == PlayPlayerKind.Human;

        private bool HasKing(sbyte kingPiece)
        {
            for (int i = 0; i < 90; i++)
            {
                if (_session.Board.GetPiece(i) == kingPiece)
                    return true;
            }

            return false;
        }

        private int ParseSearchBudget(PlayPlayerKind playerKind)
        {
            if (int.TryParse(SimulationsTextBox.Text, out int value) && value > 0)
                return playerKind is PlayPlayerKind.Traditional or PlayPlayerKind.Pikafish ? Math.Clamp(value, 1, 64) : value;

            int fallback = playerKind is PlayPlayerKind.Traditional or PlayPlayerKind.Pikafish ? _playSettings.TraditionalDepth : _playSettings.DefaultSimulations;
            SimulationsTextBox.Text = fallback.ToString();
            return fallback;
        }

        private PlayPlayerKind GetCurrentPlayerKind() => GetPlayerKind(_session.Board.IsRedTurn);

        private PlayPlayerKind GetPlayerKind(bool red)
        {
            int selectedIndex = red ? RedPlayerComboBox?.SelectedIndex ?? 0 : BlackPlayerComboBox?.SelectedIndex ?? 0;
            return selectedIndex switch
            {
                1 => PlayPlayerKind.Mcts,
                2 => PlayPlayerKind.Traditional,
                3 => PlayPlayerKind.Pikafish,
                _ => PlayPlayerKind.Human
            };
        }

        private bool NeedsPikafish() => GetPlayerKind(true) == PlayPlayerKind.Pikafish || GetPlayerKind(false) == PlayPlayerKind.Pikafish;

        private bool NeedsMcts() => GetPlayerKind(true) == PlayPlayerKind.Mcts || GetPlayerKind(false) == PlayPlayerKind.Mcts;

        private bool NeedsTraditional() => GetPlayerKind(true) == PlayPlayerKind.Traditional || GetPlayerKind(false) == PlayPlayerKind.Traditional;

        private int ResolveTraditionalRootParallelism()
        {
            return _playSettings.TraditionalRootParallelism > 0
                ? _playSettings.TraditionalRootParallelism
                : Math.Clamp(Environment.ProcessorCount, 1, 16);
        }

        private void RefreshEngineTypeUi()
        {
            bool traditional = NeedsTraditional();
            bool pikafish = NeedsPikafish();
            bool mcts = NeedsMcts();
            if (ModelFileLabel != null)
            {
                ModelFileLabel.IsEnabled = mcts || pikafish;
                ModelFileLabel.Text = pikafish ? "Pikafish executable" : "Model file";
            }
            if (ModelFilePanel != null)
                ModelFilePanel.IsEnabled = mcts || pikafish;
            if (SearchBudgetLabel != null)
                SearchBudgetLabel.Text = mcts && !traditional && !pikafish ? "Simulations" : "Depth";

            if (SimulationsTextBox != null)
                SimulationsTextBox.Text = traditional || pikafish
                    ? _playSettings.TraditionalDepth.ToString()
                    : _playSettings.DefaultSimulations.ToString();

            if (ModelPathTextBox != null)
                ModelPathTextBox.Text = FindDefaultEnginePath();
        }

        private CancellationToken ResetAiToken()
        {
            _aiCts?.Cancel();
            _aiCts?.Dispose();
            _aiCts = new CancellationTokenSource();
            return _aiCts.Token;
        }

        private string FindDefaultEnginePath()
        {
            return NeedsPikafish() ? FindDefaultPikafishPath() : FindDefaultModelPath();
        }

        private string FindDefaultPikafishPath()
        {
            if (!string.IsNullOrWhiteSpace(_playSettings.PikafishPath))
            {
                string configured = Path.GetFullPath(_playSettings.PikafishPath);
                if (File.Exists(configured))
                    return configured;
            }

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));
            string[] candidates =
            {
                Path.Combine(baseDir, "pikafish.exe"),
                Path.Combine(repoRoot, "pikafish.exe"),
                Path.Combine(repoRoot, "tools", "pikafish", "pikafish.exe"),
                Path.Combine(repoRoot, "Tools", "Pikafish", "pikafish.exe")
            };

            return candidates.FirstOrDefault(File.Exists) ?? string.Empty;
        }

        private static string FindDefaultModelPath()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));

            string[] explicitCandidates =
            {
                Path.Combine(baseDir, "data", "models", "best_model.pt"),
                Path.Combine(repoRoot, "data", "models", "best_model.pt"),
                Path.Combine(baseDir, "data", "models", "league", "agent_0.pt"),
                Path.Combine(repoRoot, "data", "models", "league", "agent_0.pt")
            };

            foreach (string candidate in explicitCandidates)
            {
                if (File.Exists(candidate))
                    return candidate;
            }

            foreach (string root in new[] { baseDir, repoRoot })
            {
                string modelsDir = Path.Combine(root, "data", "models");
                if (!Directory.Exists(modelsDir))
                    continue;

                var latest = Directory.GetFiles(modelsDir, "*.pt", SearchOption.AllDirectories)
                    .Select(path => new FileInfo(path))
                    .OrderByDescending(file => file.LastWriteTimeUtc)
                    .FirstOrDefault();

                if (latest != null)
                    return latest.FullName;
            }

            return string.Empty;
        }

        private static PlayStrengthSettings LoadPlaySettings()
        {
            foreach (string candidate in GetPlaySettingsCandidates())
            {
                if (!File.Exists(candidate))
                    continue;

                try
                {
                    string json = File.ReadAllText(candidate);
                    var settings = JsonSerializer.Deserialize<PlayStrengthSettings>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                    return PlayStrengthSettings.Sanitize(settings);
                }
                catch
                {
                    return PlayStrengthSettings.Sanitize(null);
                }
            }

            return PlayStrengthSettings.Sanitize(null);
        }

        private static IEnumerable<string> GetPlaySettingsCandidates()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));

            yield return Path.Combine(baseDir, "playsettings.json");
            yield return Path.Combine(repoRoot, "ChineseChessAI.Play", "playsettings.json");
        }

        private OpeningBook GetOrLoadOpeningBook()
        {
            if (_openingBook != null)
                return _openingBook;

            _openingBook = new OpeningBook(maxPly: 24);
            string cachePath = GetOpeningBookCachePath();
            if (_openingBook.LoadCache(cachePath))
            {
                AppendLog($"Opening book cache loaded: {cachePath}, positions={_openingBook.PositionCount}.");
                return _openingBook;
            }

            foreach (string source in GetMasterDataCandidates())
            {
                if (!Directory.Exists(source) && !File.Exists(source))
                    continue;

                int loaded = _openingBook.LoadFromPath(source);
                AppendLog($"Opening book loaded from {source}: {loaded} games.");
                if (_openingBook.PositionCount > 0)
                {
                    _openingBook.SaveCache(cachePath);
                    AppendLog($"Opening book cache saved: {cachePath}.");
                }
                break;
            }

            return _openingBook;
        }

        private static string GetOpeningBookCachePath()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));
            return Path.Combine(repoRoot, "data", "opening_book.json");
        }

        private static IEnumerable<string> GetMasterDataCandidates()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string repoRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", ".."));

            yield return Path.Combine(repoRoot, "xqdb_masters_40711_UCI_games.pgn");
            yield return Path.Combine(baseDir, "data", "master_data");
            yield return Path.Combine(repoRoot, "data", "master_data");
            yield return Path.Combine(repoRoot, "master_data");
        }

        private void AppendLog(string message)
        {
            LogBox.AppendText($"{DateTime.Now:HH:mm:ss} {message}{Environment.NewLine}");
            LogBox.ScrollToEnd();
        }

        private void DisposeResources()
        {
            _aiCts?.Cancel();
            _aiCts?.Dispose();
            _aiCts = null;
            DisposeEngine();
        }

        private void DisposeEngine()
        {
            _mctsEngine?.Dispose();
            _mctsEngine = null;
            _traditionalEngine = null;
            _pikafishEngine?.Dispose();
            _pikafishEngine = null;

            _model?.Dispose();
            _model = null;

            _loadedModelPath = string.Empty;
        }
    }
}
