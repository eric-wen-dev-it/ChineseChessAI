using ChineseChessAI.Core;
using ChineseChessAI.MCTS;

namespace ChineseChessAI.Training
{
    public sealed class MctsGameEngineAdapter : IGameEngine, IDisposable
    {
        private readonly MCTSEngine _engine;

        public MctsGameEngineAdapter(MCTSEngine engine)
        {
            _engine = engine;
        }

        public Task<(Move Move, float[] Policy)> GetMoveWithPolicyAsync(
            Board board,
            int searchBudget,
            int currentMoves,
            int maxMoves,
            CancellationToken cancellationToken)
        {
            return _engine.GetMoveWithProbabilitiesAsArrayAsync(board, searchBudget, currentMoves, maxMoves, cancellationToken);
        }

        public void NotifyMovePlayed(Board boardAfterMove, Move move)
        {
            _engine.NotifyMovePlayed(boardAfterMove, move);
        }

        public void Dispose()
        {
            _engine.Dispose();
        }
    }
}
