using System.Collections.Generic;
using ChineseChessAI.Utils;

namespace ChineseChessAI.Core
{
    /// <summary>
    /// Unified rule engine for move generation, validation, notation parsing, and immediate kill detection.
    /// </summary>
    public sealed class ChineseChessRuleEngine
    {
        private readonly MoveGenerator _generator;

        public MoveGenerator Generator => _generator;

        public ChineseChessRuleEngine(MoveGenerator? generator = null)
        {
            _generator = generator ?? new MoveGenerator();
        }

        public List<Move> GetLegalMoves(Board board, bool skipPerpetualCheck = false)
        {
            return _generator.GenerateLegalMoves(board, skipPerpetualCheck);
        }

        public string ValidateMove(Board board, Move move, bool skipPerpetualCheck = false)
        {
            return _generator.GetMoveValidationResult(board, move, skipPerpetualCheck);
        }

        public bool IsKingSafe(Board board, bool checkRed)
        {
            return _generator.IsKingSafe(board, checkRed);
        }

        public Move? GetCaptureKingMove(Board board)
        {
            return _generator.GetCaptureKingMove(board);
        }

        public bool TryResolveNotation(Board board, string rawMove, out Move move, out string normalizedUcci, out string reason, bool skipPerpetualCheck = false)
        {
            move = default;
            normalizedUcci = string.Empty;

            if (string.IsNullOrWhiteSpace(rawMove))
            {
                reason = "空着法";
                return false;
            }

            string? ucci = NotationConverter.ConvertToUcci(board, rawMove, _generator, skipPerpetualCheck);
            if (string.IsNullOrEmpty(ucci))
            {
                reason = "无法解析棋谱";
                return false;
            }

            normalizedUcci = ucci;
            return TryResolveUcci(board, ucci, out move, out reason, skipPerpetualCheck);
        }

        public bool TryResolveUcci(Board board, string ucciMove, out Move move, out string reason, bool skipPerpetualCheck = false)
        {
            move = default;
            var parsedMove = NotationConverter.UcciToMove(ucciMove);
            if (!parsedMove.HasValue)
            {
                reason = "无效UCCI";
                return false;
            }

            move = parsedMove.Value;
            reason = ValidateMove(board, move, skipPerpetualCheck);
            return reason == "合法";
        }
    }
}
