namespace ChineseChessAI.Traditional
{
    public sealed class TraditionalEngineOptions
    {
        public int MateScore { get; init; } = 1_000_000;

        public bool UseQuiescenceSearch { get; init; } = true;

        public bool SkipPerpetualCheckInsideSearch { get; init; } = true;

        public bool SkipPerpetualCheckAtRoot { get; init; } = false;

        public int TranspositionTableEntries { get; init; } = 1_048_576;

        public int MateSearchPly { get; init; } = 5;

        public bool UseNullMovePruning { get; init; } = true;

        public bool UseFutilityPruning { get; init; } = true;

        public bool UseRazoring { get; init; } = true;

        public bool UseSeePruning { get; init; } = true;

        public OpeningBook? OpeningBook { get; init; }

        public OpeningBookMode OpeningBookMode { get; init; } = OpeningBookMode.Weighted;

        public OpeningBook? MoveOrderingBook { get; init; }

        public MasterKnowledgeBook? MasterKnowledgeBook { get; init; }

        // 直接照谱出着的门槛：该着法至少出现过的大师对局数，以及
        // 从行棋方视角的最低得分率（胜=1/和=0.5，胜负未知的对局不计入）。
        public int MasterBookMinCount { get; init; } = 3;

        public double MasterBookMinWinRate { get; init; } = 0.40;

        public int RootParallelism { get; init; } = Math.Clamp(Environment.ProcessorCount, 1, 16);
    }
}
