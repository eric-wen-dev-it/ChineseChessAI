using ChineseChessAI.Core;
using ChineseChessAI.Traditional;

namespace ChineseChessAI.Training
{
    public sealed class TraditionalGameEngineAdapter : IGameEngine
    {
        private const float BestMoveProbability = 0.85f;
        private readonly TraditionalEngine _engine;
        private readonly int _quiescenceDepth;
        private readonly ChineseChessRuleEngine _rules = new ChineseChessRuleEngine();

        public TraditionalGameEngineAdapter(TraditionalEngine? engine = null, int quiescenceDepth = 4)
        {
            _engine = engine ?? new TraditionalEngine();
            _quiescenceDepth = quiescenceDepth;
        }

        public Task<(Move Move, float[] Policy)> GetMoveWithPolicyAsync(
            Board board,
            int searchBudget,
            int currentMoves,
            int maxMoves,
            CancellationToken cancellationToken)
        {
            int depth = Math.Clamp(searchBudget, 1, 12);
            int moveTimeMs = ComputeMoveTimeMs(depth);
            var policy = new float[8100];
            var legalMoves = _rules.GetLegalMoves(board, skipPerpetualCheck: false, cancellationToken: cancellationToken);
            if (legalMoves.Count == 0)
                return Task.FromResult((default(Move), policy));

            SearchResult result;
            using (var moveCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            {
                moveCts.CancelAfter(moveTimeMs + 2000);
                try
                {
                    result = _engine.Search(board, new SearchLimits(depth, moveTimeMs, _quiescenceDepth), moveCts.Token);
                }
                catch (OperationCanceledException)
                {
                    return Task.FromResult(CreateFallbackPolicy(legalMoves, policy));
                }
            }

            bool hasValidBestMove = result.BestMove.From != result.BestMove.To
                && legalMoves.Any(move => move.From == result.BestMove.From && move.To == result.BestMove.To);
            if (!hasValidBestMove)
            {
                return Task.FromResult(CreateFallbackPolicy(legalMoves, policy));
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

        public void NotifyMovePlayed(Board boardAfterMove, Move move)
        {
        }

        /// <summary>
        /// 标尺每手时限=目标深度的安全帽,按深度分档:depth≤4→75s、5→150s、6→300s、7→600s、≥8 封顶 600s
        /// (75000 × 2^(depth-4),夹在 [75s, 600s])。给足时限让高深度标尺能可靠跑到名义深度、输出真实评分,
        /// 而非卡在名义深度以下的浅层乐观分。
        /// </summary>
        public static int ComputeMoveTimeMs(int depth)
        {
            int shift = Math.Clamp(depth - 4, 0, 8);
            long ms = 75000L * (1L << shift);
            return (int)Math.Clamp(ms, 75000L, 600000L);
        }

        private static (Move Move, float[] Policy) CreateFallbackPolicy(List<Move> legalMoves, float[] policy)
        {
            float uniformProbability = 1.0f / legalMoves.Count;
            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < policy.Length)
                    policy[idx] = uniformProbability;
            }

            return (legalMoves[0], policy);
        }
    }
}
