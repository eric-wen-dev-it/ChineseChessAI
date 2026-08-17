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

        // ===== 静搜增强开关(FC 对照实验处方:静搜 > 排序/TT > 加深)=====
        // 默认全关 = 与旧版逐位同行为,严禁改默认;要用增强必须显式调 WithEnhancedQuiescence()。
        // 联赛标尺自 2026-08-24 起显式启用增强档(TrainingOrchestrator.CreateLeagueGameEngine),
        // 该次切换重开了全体 Elo 重定价窗口。

        // 静搜节点查/存置换表(深度 0 条目,不会污染主搜索的按深度探测)。
        public bool UseQuiescenceTT { get; init; }

        // Delta 剪枝:stand-pat + 被吃子价值 + 边际 仍到不了 alpha 的吃子直接跳过。
        public bool UseQuiescenceDeltaPruning { get; init; }

        public int QuiescenceDeltaMargin { get; init; } = 200;

        // 静搜内 SEE 剪枝:静态交换明亏的吃子跳过(非应将、无杀威胁时)。
        public bool UseQuiescenceSeePruning { get; init; }

        // 前多少个静搜层把"将军着法"纳入战术着法(每层需对全部合法着法
        // Push+IsChecking 逐一探测,是静搜每节点成本的大头)。int.MaxValue=
        // 旧行为(所有层都探测);增强档建议 2:近端保杀觉,远端纯吃子延伸。
        public int QuiescenceCheckPlies { get; init; } = int.MaxValue;

        // 增强档预设:在本配置基础上打开全部静搜增强。谱/TT 大小等沿用原值。
        public TraditionalEngineOptions WithEnhancedQuiescence()
        {
            return new TraditionalEngineOptions
            {
                MateScore = MateScore,
                UseQuiescenceSearch = UseQuiescenceSearch,
                SkipPerpetualCheckInsideSearch = SkipPerpetualCheckInsideSearch,
                SkipPerpetualCheckAtRoot = SkipPerpetualCheckAtRoot,
                TranspositionTableEntries = TranspositionTableEntries,
                MateSearchPly = MateSearchPly,
                UseNullMovePruning = UseNullMovePruning,
                UseFutilityPruning = UseFutilityPruning,
                UseRazoring = UseRazoring,
                UseSeePruning = UseSeePruning,
                OpeningBook = OpeningBook,
                OpeningBookMode = OpeningBookMode,
                MoveOrderingBook = MoveOrderingBook,
                MasterKnowledgeBook = MasterKnowledgeBook,
                MasterBookMinCount = MasterBookMinCount,
                MasterBookMinWinRate = MasterBookMinWinRate,
                RootParallelism = RootParallelism,
                UseQuiescenceTT = true,
                UseQuiescenceDeltaPruning = true,
                QuiescenceDeltaMargin = QuiescenceDeltaMargin,
                UseQuiescenceSeePruning = true,
                QuiescenceCheckPlies = 2
            };
        }

        public OpeningBook? OpeningBook { get; init; }

        public OpeningBookMode OpeningBookMode { get; init; } = OpeningBookMode.Weighted;

        public OpeningBook? MoveOrderingBook { get; init; }

        public MasterKnowledgeBook? MasterKnowledgeBook { get; init; }

        // 直接照谱出着的门槛：该着法至少出现过的大师对局数，以及
        // 从行棋方视角的最低得分率（胜=1/和=0.5，胜负未知的对局不计入）。
        public int MasterBookMinCount { get; init; } = 3;

        public double MasterBookMinWinRate { get; init; } = 0.40;

        // 默认单线程。Lazy SMP(RootParallelism>1)经 2026-08-24 实测在本引擎上严格
        // 劣于单线程(同局面到 depth9:单线程 60s/2.7M 节点 vs SMP 120s/29M 节点仍未
        // 完成),单节点过重导致多线程净负;设 >1 仅供未来 SMP 调优显式启用。
        public int RootParallelism { get; init; } = 1;
    }
}
