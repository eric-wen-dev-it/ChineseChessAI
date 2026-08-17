using ChineseChessAI.Core;
using System.Diagnostics;

namespace ChineseChessAI.Traditional
{
    public sealed class TraditionalEngine
    {
        private readonly TraditionalSearch _search;
        private readonly TraditionalEngineOptions _options;
        private readonly MoveGenerator _generator;
        private readonly TraditionalMoveOrdering _moveOrdering;
        private readonly TranspositionTable _table;

        public TraditionalEngine(TraditionalEngineOptions? options = null, MoveGenerator? generator = null)
        {
            _options = options ?? new TraditionalEngineOptions();
            _generator = generator ?? new MoveGenerator();
            var evaluator = new TraditionalEvaluator();
            _moveOrdering = new TraditionalMoveOrdering(_generator, _options.MoveOrderingBook ?? _options.OpeningBook, _options.MasterKnowledgeBook);
            _table = new TranspositionTable(_options.TranspositionTableEntries);
            _search = new TraditionalSearch(_generator, evaluator, _moveOrdering, _options, _table);
        }

        public SearchResult Search(Board board, SearchLimits limits, CancellationToken cancellationToken = default)
        {
            // 优先查带胜负统计的知识谱（覆盖 120 步），按胜率下界选着；
            // 该谱查不到时再退回按流行度统计的开局谱（覆盖 24 步）。
            if (_options.MasterKnowledgeBook != null && _options.MasterKnowledgeBook.TryGetBookMove(
                    board,
                    _options.OpeningBookMode,
                    _options.MasterBookMinCount,
                    _options.MasterBookMinWinRate,
                    out var masterKnowledge))
            {
                var masterMove = masterKnowledge.Move;
                return new SearchResult(masterMove, 0, 0, 0, TimeSpan.Zero, new[] { masterMove }, true, FromBook: true);
            }

            if (_options.OpeningBook != null && _options.OpeningBook.TryGetMove(board, _options.OpeningBookMode, out var bookMove, _options.MasterBookMinCount))
            {
                return new SearchResult(bookMove, 0, 0, 0, TimeSpan.Zero, new[] { bookMove }, true, FromBook: true);
            }

            if (ShouldUseLazySmp(limits))
                return SearchLazySmp(board, limits, cancellationToken);

            return _search.Search(board, limits, cancellationToken);
        }

        private bool ShouldUseLazySmp(SearchLimits limits)
        {
            // 仅限时搜索(Play/FC 桥接)用多线程;联赛用 FixedDepth(MoveTimeMs=0),
            // 走单线程,棋力不受此路径影响。
            return _options.RootParallelism > 1 && limits.MoveTimeMs > 0 && limits.MaxDepth >= 3;
        }

        // Lazy SMP:N 个搜索线程在同一局面上各自跑迭代加深,共享唯一一张无锁置换表。
        // 相比旧"并行根"(每个根着法独立全窗口 + 各自置换表,实测 40s/6.1M 节点仅到
        // depth4、且输出浅层乐观假分)——本方案让线程经共享 TT 交叉授粉,α-β 剪枝正常
        // 生效,取完成深度最深的结果。奇偶线程关/开空着裁剪制造搜索多样性,避免多核
        // 锁步重复同一棵树。
        private SearchResult SearchLazySmp(Board board, SearchLimits limits, CancellationToken cancellationToken)
        {
            int threads = Math.Clamp(_options.RootParallelism, 1, 16);
            if (threads <= 1)
                return _search.Search(board, limits, cancellationToken);

            var searches = new TraditionalSearch[threads];
            var boards = new Board[threads];
            searches[0] = _search;
            boards[0] = board.Clone();
            for (int i = 1; i < threads; i++)
            {
                searches[i] = CreateSharedSearch(DiversifyOptions(i));
                boards[i] = board.Clone();
            }

            var results = new SearchResult?[threads];
            try
            {
                // 故意不给 ParallelOptions 传 CancellationToken:协作取消会让 Parallel
                // 在超时瞬间抛 OCE 并等待全部副本汇合,该汇合在 net10 上出现过挂死
                // (2026-08-17 实测)。取消一律由各线程 Search 内部 ShouldStop 处理,
                // 其内部已捕获 OCE 并返回可用结果,不会向外抛。
                Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, i =>
                {
                    try
                    {
                        // 主线程(0)恒从 depth1 起,保证任何时刻都有可用最佳着法;
                        // 辅助线程错位起始深度(1/2/3 轮转),跳过浅层重复迭代,经共享
                        // 置换表让主线程飞过浅层、整体更快抵达深层。
                        int startDepth = i == 0 ? 1 : 1 + (i % 3);
                        results[i] = searches[i].Search(boards[i], limits, cancellationToken, null, null, startDepth);
                    }
                    catch (Exception)
                    {
                        results[i] = null;
                    }
                });
            }
            catch (Exception)
            {
                // 任何并行执行期异常只作废多线程结果,退回主线程结论。
            }

            long totalNodes = 0;
            SearchResult? best = null;
            for (int i = 0; i < threads; i++)
            {
                var r = results[i];
                if (r == null)
                    continue;
                totalNodes += r.Nodes;
                if (best == null
                    || r.Depth > best.Depth
                    || (r.Depth == best.Depth && r.Completed && !best.Completed))
                {
                    best = r;
                }
            }

            if (best == null)
                return _search.Search(board, limits, cancellationToken);

            // 汇总全线程节点数供日志如实反映总工作量。
            return best with { Nodes = totalNodes };
        }

        private TraditionalSearch CreateSharedSearch(TraditionalEngineOptions options)
        {
            var generator = new MoveGenerator();
            var evaluator = new TraditionalEvaluator();
            var moveOrdering = new TraditionalMoveOrdering(generator, options.MoveOrderingBook ?? options.OpeningBook, options.MasterKnowledgeBook);
            return new TraditionalSearch(generator, evaluator, moveOrdering, options, _table);
        }

        // 为 Lazy SMP 辅助线程派生一个仅微调裁剪档的选项副本(共享同一批只读谱)。
        // 奇数号线程关空着裁剪 → 更保守、对战术更敏感,与主线程走不同的树;并列深度
        // 时择优仍以主线程为先,故对最终选着安全。
        private TraditionalEngineOptions DiversifyOptions(int threadIndex)
        {
            bool useNullMove = (threadIndex % 2 == 0) && _options.UseNullMovePruning;
            return new TraditionalEngineOptions
            {
                MateScore = _options.MateScore,
                UseQuiescenceSearch = _options.UseQuiescenceSearch,
                SkipPerpetualCheckInsideSearch = _options.SkipPerpetualCheckInsideSearch,
                SkipPerpetualCheckAtRoot = _options.SkipPerpetualCheckAtRoot,
                TranspositionTableEntries = _options.TranspositionTableEntries,
                MateSearchPly = _options.MateSearchPly,
                UseNullMovePruning = useNullMove,
                UseFutilityPruning = _options.UseFutilityPruning,
                UseRazoring = _options.UseRazoring,
                UseSeePruning = _options.UseSeePruning,
                UseQuiescenceTT = _options.UseQuiescenceTT,
                UseQuiescenceDeltaPruning = _options.UseQuiescenceDeltaPruning,
                QuiescenceDeltaMargin = _options.QuiescenceDeltaMargin,
                UseQuiescenceSeePruning = _options.UseQuiescenceSeePruning,
                QuiescenceCheckPlies = _options.QuiescenceCheckPlies,
                OpeningBook = null,
                OpeningBookMode = OpeningBookMode.Off,
                MoveOrderingBook = _options.MoveOrderingBook,
                MasterKnowledgeBook = _options.MasterKnowledgeBook,
                MasterBookMinCount = _options.MasterBookMinCount,
                MasterBookMinWinRate = _options.MasterBookMinWinRate,
                RootParallelism = 1
            };
        }
    }
}
