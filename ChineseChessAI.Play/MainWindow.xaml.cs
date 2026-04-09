using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ChineseChessAI.Play
{
    public partial class MainWindow : Window
    {
        private const int DefaultMaxMoves = 150;
        private const int DefaultBatchSize = 32;

        private readonly Button[] _cellButtons = new Button[90];
        private readonly GameRuleSession _session = new();
        private readonly ChineseChessRuleEngine _rules = new();
        private readonly HashSet<int> _legalTargets = new();

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

            ModelPathTextBox.Text = FindDefaultModelPath();
            RefreshBoard();
            UpdateUiState("Choose a model and start a game.");

            Closed += (_, _) => DisposeResources();
        }

        private void InitializeBoard()
        {
            BoardGrid.Children.Clear();

            for (int i = 0; i < _cellButtons.Length; i++)
            {
                var button = new Button
                {
                    Tag = i,
                    FontSize = 28,
                    FontFamily = new FontFamily("KaiTi"),
                    FontWeight = FontWeights.Bold,
                    Margin = new Thickness(1),
                    BorderBrush = new SolidColorBrush(Color.FromRgb(121, 79, 35)),
                    BorderThickness = new Thickness(1),
                    Background = GetDefaultCellBrush(i),
                    Foreground = Brushes.Black,
                    FocusVisualStyle = null
                };
                button.Click += OnBoardCellClick;
                _cellButtons[i] = button;
                BoardGrid.Children.Add(button);
            }
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

            int index = (int)((Button)sender).Tag;
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
                    token);

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
                if (!_gameOver)
                    UpdateUiState(IsHumanTurn() ? "Your move." : "AI ready.");
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
                UpdateUiState("Black wins.");
                AppendLog("Game over: Black wins.");
                return true;
            }

            if (!HasKing(-1))
            {
                _gameOver = true;
                UpdateUiState("Red wins.");
                AppendLog("Game over: Red wins.");
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
                return true;
            }

            return false;
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
                _engine = new MCTSEngine(model, DefaultBatchSize);
                _loadedModelPath = fullPath;

                AppendLog($"Loaded model: {Path.GetFileName(fullPath)}");
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
            for (int i = 0; i < _cellButtons.Length; i++)
            {
                var button = _cellButtons[i];
                sbyte piece = _session.Board.GetPiece(i);

                button.Content = piece == 0 ? string.Empty : Board.GetPieceName(piece);
                button.Foreground = piece > 0 ? Brushes.Firebrick : Brushes.Black;
                button.Background = GetCellBrush(i);
            }

            MoveHistoryTextBox.Text = _session.Board.GetMoveHistoryString();
            BoardGrid.IsEnabled = !_gameOver && !_isAiThinking && IsHumanTurn();
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
        }

        private Brush GetCellBrush(int index)
        {
            if (_selectedIndex == index)
                return new SolidColorBrush(Color.FromRgb(240, 183, 79));

            if (_legalTargets.Contains(index))
                return new SolidColorBrush(Color.FromRgb(243, 229, 152));

            if (_session.Board.LastMove is Move lastMove && (lastMove.From == index || lastMove.To == index))
                return new SolidColorBrush(Color.FromRgb(231, 209, 162));

            return GetDefaultCellBrush(index);
        }

        private static Brush GetDefaultCellBrush(int index)
        {
            int row = index / 9;
            int col = index % 9;
            return ((row + col) & 1) == 0
                ? new SolidColorBrush(Color.FromRgb(249, 233, 191))
                : new SolidColorBrush(Color.FromRgb(242, 222, 174));
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

            SimulationsTextBox.Text = "400";
            return 400;
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

            var explicitCandidates = new[]
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
