using ChineseChessAI.Core;

namespace ChineseChessAI.Traditional
{
    public sealed record SearchResult(
        Move BestMove,
        int Score,
        int Depth,
        long Nodes,
        TimeSpan Elapsed,
        IReadOnlyList<Move> PrincipalVariation,
        bool Completed);
}
