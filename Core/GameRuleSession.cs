using System.Collections.Generic;
using ChineseChessAI.Utils;

namespace ChineseChessAI.Core
{
    /// <summary>
    /// 状态化规则会话：持有当前棋盘，并提供统一的棋谱解析、合法性校验与落子入口。
    /// </summary>
    public sealed class GameRuleSession
    {
        private readonly MoveGenerator _generator;
        private readonly List<Move> _moveHistory = new();
        private readonly List<string> _ucciHistory = new();

        public Board Board { get; }
        public MoveGenerator Generator => _generator;
        public IReadOnlyList<Move> MoveHistory => _moveHistory;
        public IReadOnlyList<string> UcciHistory => _ucciHistory;

        public GameRuleSession(MoveGenerator? generator = null)
        {
            Board = new Board();
            _generator = generator ?? new MoveGenerator();
        }

        public void Reset()
        {
            Board.Reset();
            _moveHistory.Clear();
            _ucciHistory.Clear();
        }

        public List<Move> GetLegalMoves(bool skipPerpetualCheck = false)
        {
            return _generator.GenerateLegalMoves(Board, skipPerpetualCheck);
        }

        public string ValidateMove(Move move, bool skipPerpetualCheck = false)
        {
            return _generator.GetMoveValidationResult(Board, move, skipPerpetualCheck);
        }

        public bool TryResolveNotation(string rawMove, out Move move, out string normalizedUcci, out string reason, bool skipPerpetualCheck = false)
        {
            move = default;
            normalizedUcci = string.Empty;

            if (string.IsNullOrWhiteSpace(rawMove))
            {
                reason = "空着法";
                return false;
            }

            string? ucci = NotationConverter.ConvertToUcci(Board, rawMove, _generator, skipPerpetualCheck);
            if (string.IsNullOrEmpty(ucci))
            {
                reason = "无法解析棋谱";
                return false;
            }

            normalizedUcci = ucci;
            return TryResolveUcci(ucci, out move, out reason, skipPerpetualCheck);
        }

        public bool TryResolveUcci(string ucciMove, out Move move, out string reason, bool skipPerpetualCheck = false)
        {
            move = default;
            var parsedMove = NotationConverter.UcciToMove(ucciMove);
            if (!parsedMove.HasValue)
            {
                reason = "无效UCCI";
                return false;
            }

            move = parsedMove.Value;
            reason = ValidateMove(move, skipPerpetualCheck);
            return reason == "合法";
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
