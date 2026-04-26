using ChineseChessAI.Core;

namespace ChineseChessAI.Traditional
{
    public sealed class TraditionalMoveOrdering
    {
        private static readonly int[] VictimValues = { 0, 10_000, 200, 200, 400, 900, 450, 100 };
        private readonly MoveGenerator _generator;
        private readonly OpeningBook? _book;
        private readonly MasterKnowledgeBook? _knowledgeBook;

        public TraditionalMoveOrdering(MoveGenerator generator, OpeningBook? book = null, MasterKnowledgeBook? knowledgeBook = null)
        {
            _generator = generator;
            _book = book;
            _knowledgeBook = knowledgeBook;
        }

        public List<Move> OrderMoves(
            Board board,
            List<Move> moves,
            Move? preferredMove = null,
            Move? killerOne = null,
            Move? killerTwo = null,
            int[]? history = null)
        {
            return moves
                .Select(move => (Move: move, Score: ScoreMove(board, move, preferredMove, killerOne, killerTwo, history)))
                .OrderByDescending(x => x.Score)
                .Select(x => x.Move)
                .ToList();
        }

        private int ScoreMove(Board board, Move move, Move? preferredMove, Move? killerOne, Move? killerTwo, int[]? history)
        {
            if (preferredMove.HasValue && move.Equals(preferredMove.Value))
                return 1_000_000;

            int score = 0;
            sbyte attacker = board.GetPiece(move.From);
            sbyte victim = board.GetPiece(move.To);
            if (_book != null)
                score += _book.GetMoveOrderingBonus(board, move);
            if (_knowledgeBook != null)
                score += _knowledgeBook.GetMoveOrderingBonus(board, move);

            if (victim != 0)
            {
                score += 100_000;
                score += VictimValues[Math.Abs(victim)] * 16 - VictimValues[Math.Abs(attacker)];
            }
            else
            {
                if (killerOne.HasValue && move.Equals(killerOne.Value))
                    score += 80_000;
                else if (killerTwo.HasValue && move.Equals(killerTwo.Value))
                    score += 70_000;

                if (history != null)
                    score += Math.Min(60_000, history[move.ToNetworkIndex()]);
            }

            board.Push(move.From, move.To);
            try
            {
                if (_generator.IsChecking(board, !board.IsRedTurn))
                    score += 20_000;
            }
            finally
            {
                board.Pop();
            }

            score += CenterBonus(move.To);
            return score;
        }

        private static int CenterBonus(int index)
        {
            int col = index % 9;
            return 8 - Math.Abs(col - 4);
        }
    }
}
