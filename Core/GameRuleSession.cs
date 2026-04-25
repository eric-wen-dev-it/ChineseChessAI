using ChineseChessAI.Utils;

namespace ChineseChessAI.Core
{
    /// <summary>
    /// Stateful rule session that tracks a single board and its applied move history.
    /// </summary>
    public sealed class GameRuleSession
    {
        private readonly ChineseChessRuleEngine _rules;
        private readonly List<Move> _moveHistory = new();
        private readonly List<string> _ucciHistory = new();

        public Board Board
        {
            get;
        }
        public ChineseChessRuleEngine Rules => _rules;
        public IReadOnlyList<Move> MoveHistory => _moveHistory;
        public IReadOnlyList<string> UcciHistory => _ucciHistory;

        public GameRuleSession(ChineseChessRuleEngine? rules = null)
        {
            Board = new Board();
            _rules = rules ?? new ChineseChessRuleEngine();
        }

        public void Reset()
        {
            Board.Reset();
            _moveHistory.Clear();
            _ucciHistory.Clear();
        }

        public List<Move> GetLegalMoves(bool skipPerpetualCheck = false)
        {
            return _rules.GetLegalMoves(Board, skipPerpetualCheck);
        }

        public string ValidateMove(Move move, bool skipPerpetualCheck = false)
        {
            return _rules.ValidateMove(Board, move, skipPerpetualCheck);
        }

        public bool TryResolveNotation(string rawMove, out Move move, out string normalizedUcci, out string reason, bool skipPerpetualCheck = false)
        {
            return _rules.TryResolveNotation(Board, rawMove, out move, out normalizedUcci, out reason, skipPerpetualCheck);
        }

        public bool TryResolveUcci(string ucciMove, out Move move, out string reason, bool skipPerpetualCheck = false)
        {
            return _rules.TryResolveUcci(Board, ucciMove, out move, out reason, skipPerpetualCheck);
        }

        public bool TryApplyNotation(string rawMove, out Move move, out string normalizedUcci, out string reason, bool skipPerpetualCheck = false)
        {
            if (!TryResolveNotation(rawMove, out move, out normalizedUcci, out reason, skipPerpetualCheck))
                return false;

            ApplyMove(move, normalizedUcci);
            return true;
        }

        public bool TryApplyUcci(string ucciMove, out Move move, out string reason, bool skipPerpetualCheck = false)
        {
            if (!TryResolveUcci(ucciMove, out move, out reason, skipPerpetualCheck))
                return false;

            ApplyMove(move, NotationConverter.MoveToUcci(move));
            return true;
        }

        public bool TryApplyMove(Move move, out string reason, bool skipPerpetualCheck = false)
        {
            reason = ValidateMove(move, skipPerpetualCheck);
            if (reason != "合法")
                return false;

            ApplyMove(move);
            return true;
        }

        public void ApplyMove(Move move, string? normalizedUcci = null)
        {
            Board.Push(move.From, move.To);
            _moveHistory.Add(move);
            _ucciHistory.Add(normalizedUcci ?? NotationConverter.MoveToUcci(move));
        }

        public bool UndoLastMove()
        {
            if (_moveHistory.Count == 0)
                return false;

            Board.Pop();
            _moveHistory.RemoveAt(_moveHistory.Count - 1);
            _ucciHistory.RemoveAt(_ucciHistory.Count - 1);
            return true;
        }
    }
}
