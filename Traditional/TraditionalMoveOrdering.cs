using ChineseChessAI.Core;

namespace ChineseChessAI.Traditional
{
    public sealed class TraditionalMoveOrdering
    {
        private static readonly int[] VictimValues = { 0, 10_000, 200, 200, 450, 900, 400, 100 };
        private readonly OpeningBook? _book;
        private readonly MasterKnowledgeBook? _knowledgeBook;

        public TraditionalMoveOrdering(MoveGenerator generator, OpeningBook? book = null, MasterKnowledgeBook? knowledgeBook = null)
        {
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
            if (moves.Count <= 1)
                return moves;

            // 谱引导只在开局段的局面命中:整个局面的着法表一次取出,中后盘
            // 局面这两次字典查找即失配,循环体内零成本。
            Dictionary<Move, int>? bookMoves = null;
            _book?.TryGetPositionMoves(board.CurrentHash, out bookMoves);
            Dictionary<Move, MasterKnowledgeBook.MasterMoveKnowledge>? knowledgeMoves = null;
            _knowledgeBook?.TryGetPositionMoves(board.CurrentHash, out knowledgeMoves);

            var ordered = moves.ToArray();
            var negatedScores = new int[ordered.Length];
            for (int i = 0; i < ordered.Length; i++)
                negatedScores[i] = -ScoreMove(board, ordered[i], preferredMove, killerOne, killerTwo, history, bookMoves, knowledgeMoves);

            Array.Sort(negatedScores, ordered);
            return new List<Move>(ordered);
        }

        private static int ScoreMove(
            Board board,
            Move move,
            Move? preferredMove,
            Move? killerOne,
            Move? killerTwo,
            int[]? history,
            Dictionary<Move, int>? bookMoves,
            Dictionary<Move, MasterKnowledgeBook.MasterMoveKnowledge>? knowledgeMoves)
        {
            if (preferredMove.HasValue && move.Equals(preferredMove.Value))
                return 1_000_000;

            int score = 0;
            sbyte attacker = board.GetPiece(move.From);
            sbyte victim = board.GetPiece(move.To);
            int guidanceBonus = 0;
            if (bookMoves != null && bookMoves.TryGetValue(move, out int bookCount))
                guidanceBonus += OpeningBook.OrderingBonusForCount(bookCount);
            if (knowledgeMoves != null && knowledgeMoves.TryGetValue(move, out var knowledge))
                guidanceBonus += knowledge.OrderingBonus;
            score += Math.Min(50_000, guidanceBonus);

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
