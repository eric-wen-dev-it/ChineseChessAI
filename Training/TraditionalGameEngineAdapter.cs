using ChineseChessAI.Core;
using ChineseChessAI.Traditional;

namespace ChineseChessAI.Training
{
    public sealed class TraditionalGameEngineAdapter : IGameEngine
    {
        private const float BestMoveProbability = 0.85f;
        private readonly TraditionalEngine _engine;
        private readonly ChineseChessRuleEngine _rules = new ChineseChessRuleEngine();

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
            var legalMoves = _rules.GetLegalMoves(board, cancellationToken: cancellationToken);
            if (legalMoves.Count == 0)
                return Task.FromResult((result.BestMove, policy));

            bool hasValidBestMove = result.BestMove.From != result.BestMove.To
                && legalMoves.Any(move => move.From == result.BestMove.From && move.To == result.BestMove.To);
            if (!hasValidBestMove)
            {
                float uniformProbability = 1.0f / legalMoves.Count;
                foreach (var move in legalMoves)
                {
                    int idx = move.ToNetworkIndex();
                    if (idx >= 0 && idx < policy.Length)
                        policy[idx] = uniformProbability;
                }

                return Task.FromResult((legalMoves[0], policy));
            }

            float backgroundProbability = legalMoves.Count > 0 ? (1.0f - BestMoveProbability) / legalMoves.Count : 0.0f;
            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < policy.Length)
                    policy[idx] = backgroundProbability;
            }

            if (result.BestMove.From != result.BestMove.To)
            {
                int bestIdx = result.BestMove.ToNetworkIndex();
                if (bestIdx >= 0 && bestIdx < policy.Length)
                    policy[bestIdx] += BestMoveProbability;
            }

            return Task.FromResult((result.BestMove, policy));
        }
    }
}
