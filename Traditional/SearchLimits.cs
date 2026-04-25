namespace ChineseChessAI.Traditional
{
    public readonly record struct SearchLimits(
        int MaxDepth = 5,
        int MoveTimeMs = 1000,
        int QuiescenceDepth = 6)
    {
        public static SearchLimits FixedDepth(int depth) => new(depth, 0, 6);

        public static SearchLimits FixedTime(int milliseconds, int maxDepth = 64) => new(maxDepth, milliseconds, 6);
    }
}
