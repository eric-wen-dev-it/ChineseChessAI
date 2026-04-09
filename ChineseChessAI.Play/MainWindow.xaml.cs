using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace ChineseChessAI.Play
{
    public partial class MainWindow : Window
    {
        private const int DefaultMaxMoves = 150;

        private readonly Button[] _cellButtons = new Button[90];
        private readonly GameRuleSession _session = new();
        private readonly ChineseChessRuleEngine _rules = new();
        private readonly HashSet<int> _legalTargets = new();
        private readonly PlayStrengthSettings _playSettings = LoadPlaySettings();

        private CChessNet? _model;
        private MCTSEngine? _engine;
        private CancellationTokenSource? _aiCts;

        private int? _selectedIndex;
        private string _loadedModelPath = string.Empty;
        private bool _humanPlaysRed = true;
        private bool _isAiThinking;
        private bool _gameOver;

        public MainWindow()
        {
            InitializeComponent();
            InitializeBoard();
            Loaded += (_, _) => DrawBoardLines();

            SimulationsTextBox.Text = _playSettings.DefaultSimulations.ToString();
            ModelPathTextBox.Text = FindDefaultModelPath();
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

            if (!TryCreateEngine())
                return;

            _humanPlaysRed = HumanSideComboBox.SelectedIndex == 0;
            _session.Reset();
            _gameOver = false;
            ResetSelection();
            LogBox.Clear();

            AppendLog($"New game. Human plays {(_humanPlaysRed ? "Red" : "Black")}.");
            RefreshBoard();
            UpdateUiState("Game started.");

            if (!IsHumanTurn())
                await MakeAiMoveAsync();
        }

        private async void OnAiMoveClick(object sender, RoutedEventArgs e)
        {
            if (_gameOver || _isAiThinking)
                return;

            if (!TryCreateEngine())
                return;

            if (IsHumanTurn())
            {
                UpdateUiState("It is currently your turn.");
                return;
            }

            await MakeAiMoveAsync();
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

            string winner = _humanPlaysRed ? "Black" : "Red";
            UpdateUiState($"{winner} wins by resignation.");
            AppendLog($"Game over: {winner} wins by resignation.");
            RefreshBoard();
            ShowGameOverDialog(winner, "resignation");
        }

        private void OnBrowseModelClick(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select model file",
                Filter = "Torch model (*.pt)|*.pt|All files (*.*)|*.*"
            };

            if (dialog.ShowDialog() == true)
                ModelPathTextBox.Text = dialog.FileName;
        }

        private async void OnBoardCellClick(object sender, RoutedEventArgs e)
        {
            if (_gameOver || _isAiThinking || !IsHumanTurn())
                return;

            if (sender is not Button button || button.DataContext is not int index)
                return;

            sbyte piece = _session.Board.GetPiece(index);
            bool isOwnPiece = piece != 0 && ((piece > 0) == _humanPlaysRed);

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
                    await MakeAiMoveAsync();
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

        private async Task MakeAiMoveAsync()
        {
            if (_engine == null)
                return;

            int simulations = ParseSimulationCount();
            var token = ResetAiToken();

            _isAiThinking = true;
            ResetSelection();
            RefreshBoard();
            UpdateUiState($"AI thinking with {simulations} simulations...");

            try
            {
                var (move, _) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(
                    _session.Board,
                    simulations,
                    _session.MoveHistory.Count,
                    DefaultMaxMoves,
                    token,
                    addRootNoise: _playSettings.AddRootNoise);

                ApplyMove(move, "AI");
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
                UpdateUiState(_gameOver ? StatusTextBlock.Text : (IsHumanTurn() ? "Your move." : "AI ready."));
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
            bool humanWon = string.Equals(winner, _humanPlaysRed ? "Red" : "Black", StringComparison.Ordinal);
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

        private bool TryCreateEngine()
        {
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

            if (_engine != null && string.Equals(_loadedModelPath, fullPath, StringComparison.OrdinalIgnoreCase))
                return true;

            DisposeEngine();

            try
            {
                var model = new CChessNet();
                ModelManager.LoadModel(model, fullPath);

                _model = model;
                _engine = new MCTSEngine(model, _playSettings.BatchSize, _playSettings.CPuct);
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
            return _humanPlaysRed ? boardIndex : 89 - boardIndex;
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
            SideTextBlock.Text = $"You: {(_humanPlaysRed ? "Red" : "Black")}";

            NewGameButton.IsEnabled = !_isAiThinking;
            AiMoveButton.IsEnabled = !_gameOver && !_isAiThinking;
            ResignButton.IsEnabled = !_gameOver && !_isAiThinking;
        }

        private void ResetSelection()
        {
            _selectedIndex = null;
            _legalTargets.Clear();
        }

        private bool IsHumanTurn() => _session.Board.IsRedTurn == _humanPlaysRed;

        private bool HasKing(sbyte kingPiece)
        {
            for (int i = 0; i < 90; i++)
            {
                if (_session.Board.GetPiece(i) == kingPiece)
                    return true;
            }

            return false;
        }

        private int ParseSimulationCount()
        {
            if (int.TryParse(SimulationsTextBox.Text, out int value) && value > 0)
                return value;

            SimulationsTextBox.Text = _playSettings.DefaultSimulations.ToString();
            return _playSettings.DefaultSimulations;
        }

        private CancellationToken ResetAiToken()
        {
            _aiCts?.Cancel();
            _aiCts?.Dispose();
            _aiCts = new CancellationTokenSource();
            return _aiCts.Token;
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
            _engine?.Dispose();
            _engine = null;

            _model?.Dispose();
            _model = null;

            _loadedModelPath = string.Empty;
        }
    }
}
