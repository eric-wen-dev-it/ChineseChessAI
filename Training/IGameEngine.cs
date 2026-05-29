using ChineseChessAI.Core;

namespace ChineseChessAI.Training
{
    public interface IGameEngine
    {
        /// <param name="searchBudget">
        /// Engine-specific search budget: simulations for MCTS engines, depth for traditional search engines.
        /// </param>
        Task<(Move Move, float[] Policy)> GetMoveWithPolicyAsync(
            Board board,
            int searchBudget,
            int currentMoves,
            int maxMoves,
            CancellationToken cancellationToken);
    }
}
