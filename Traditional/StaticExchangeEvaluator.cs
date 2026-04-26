using ChineseChessAI.Core;

namespace ChineseChessAI.Traditional
{
    internal static class StaticExchangeEvaluator
    {
        private const int MaxSeeDepth = 4;
        private static readonly int[] PieceValues = { 0, 10_000, 200, 200, 400, 900, 450, 100 };

        public static int Evaluate(Board board, Move firstMove, MoveGenerator generator)
        {
            sbyte victim = board.GetPiece(firstMove.To);
            if (victim == 0)
                return 0;

            int initialGain = ValueOf(victim);
            board.Push(firstMove.From, firstMove.To);
            try
            {
                return initialGain - BestRecaptureGain(board, firstMove.To, generator, 1);
            }
            finally
            {
                board.Pop();
            }
        }

        private static int BestRecaptureGain(Board board, int target, MoveGenerator generator, int depth)
        {
            if (depth >= MaxSeeDepth)
                return 0;

            sbyte occupied = board.GetPiece(target);
            if (occupied == 0)
                return 0;

            int capturedValue = ValueOf(occupied);
            int bestGain = 0;
            foreach (var recapture in GetLeastValuableRecaptures(board, target, generator))
            {
                board.Push(recapture.From, recapture.To);
                try
                {
                    int gain = capturedValue - BestRecaptureGain(board, target, generator, depth + 1);
                    if (gain > bestGain)
                        bestGain = gain;
                }
                finally
                {
                    board.Pop();
                }
            }

            return bestGain;
        }

        private static IEnumerable<Move> GetLeastValuableRecaptures(Board board, int target, MoveGenerator generator)
        {
            return generator
                .GenerateLegalMoves(board, skipPerpetualCheck: true)
                .Where(move => move.To == target)
                .OrderBy(move => ValueOf(board.GetPiece(move.From)));
        }

        private static int ValueOf(sbyte piece)
        {
            return PieceValues[Math.Abs(piece)];
        }
    }
}
