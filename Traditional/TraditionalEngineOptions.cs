namespace ChineseChessAI.Traditional
{
    public sealed class TraditionalEngineOptions
    {
        public int MateScore { get; init; } = 1_000_000;

        public bool UseQuiescenceSearch { get; init; } = true;

        public bool SkipPerpetualCheckInsideSearch { get; init; } = true;

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

        public int RootParallelism { get; init; } = Math.Clamp(Environment.ProcessorCount, 1, 16);
    }
}
