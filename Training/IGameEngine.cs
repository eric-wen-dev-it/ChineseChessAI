using ChineseChessAI.Core;

namespace ChineseChessAI.Training
{
    public interface IGameEngine
    {
        Task<(Move Move, float[] Policy)> GetMoveWithPolicyAsync(
            Board board,
            int searchBudget,
            int currentMoves,
            int maxMoves,
            CancellationToken cancellationToken);
    }
}
