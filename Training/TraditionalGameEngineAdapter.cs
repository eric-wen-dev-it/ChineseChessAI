using ChineseChessAI.Core;
using ChineseChessAI.Traditional;

namespace ChineseChessAI.Training
{
    public sealed class TraditionalGameEngineAdapter : IGameEngine
    {
        private readonly TraditionalEngine _engine;

        public TraditionalGameEngineAdapter(TraditionalEngine? engine = null)
        {
            _engine = engine ?? new TraditionalEngine();
        }

        public Task<(Move Move, float[] Policy)> GetMoveWithPolicyAsync(
            Board board,
            int searchBudget,
            int currentMoves,
            int maxMoves,
            CancellationToken cancellationToken)
        {
            int depth = Math.Clamp(searchBudget, 1, 12);
            var result = _engine.Search(board, SearchLimits.FixedDepth(depth), cancellationToken);
            var policy = new float[8100];
            if (result.BestMove.From != result.BestMove.To)
                policy[result.BestMove.ToNetworkIndex()] = 1.0f;

            return Task.FromResult((result.BestMove, policy));
        }
    }
}
