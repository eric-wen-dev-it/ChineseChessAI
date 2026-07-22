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

        public TraditionalEngine(TraditionalEngineOptions? options = null, MoveGenerator? generator = null)
        {
            _options = options ?? new TraditionalEngineOptions();
            _generator = generator ?? new MoveGenerator();
            var evaluator = new TraditionalEvaluator();
            _moveOrdering = new TraditionalMoveOrdering(_generator, _options.MoveOrderingBook ?? _options.OpeningBook, _options.MasterKnowledgeBook);
            var table = new TranspositionTable(_options.TranspositionTableEntries);
            _search = new TraditionalSearch(_generator, evaluator, _moveOrdering, _options, table);
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

            if (ShouldUseParallelRoot(limits))
                return SearchParallelRoot(board, limits, cancellationToken);

            return _search.Search(board, limits, cancellationToken);
        }

        private bool ShouldUseParallelRoot(SearchLimits limits)
        {
            return _options.RootParallelism > 1 && limits.MoveTimeMs > 0 && limits.MaxDepth >= 3;
        }

        private SearchResult SearchParallelRoot(Board board, SearchLimits limits, CancellationToken cancellationToken)
        {
            var stopwatch = Stopwatch.StartNew();
            var rootMoves = _generator.GenerateLegalMoves(board, skipPerpetualCheck: _options.SkipPerpetualCheckAtRoot);
            if (rootMoves.Count == 0)
                return new SearchResult(default, -_options.MateScore, 0, 0, stopwatch.Elapsed, Array.Empty<Move>(), true);

            var orderedMoves = _moveOrdering.OrderMoves(board, rootMoves);
            if (orderedMoves.Count == 1)
                return new SearchResult(orderedMoves[0], 0, 0, 0, stopwatch.Elapsed, new[] { orderedMoves[0] }, true);

            using var timeCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeCts.CancelAfter(limits.MoveTimeMs);
            var token = timeCts.Token;

            // 每个根着法一个持久工人:置换表跨轮复用,重搜浅层几乎免费。
            var workers = new TraditionalEngine[orderedMoves.Count];
            var childBoards = new Board[orderedMoves.Count];
            for (int i = 0; i < orderedMoves.Count; i++)
            {
                workers[i] = CreateWorkerWithoutBookMove();
                var childBoard = board.Clone();
                childBoard.Push(orderedMoves[i].From, orderedMoves[i].To);
                childBoards[i] = childBoard;
            }

            object sync = new();
            long totalNodes = 0;
            Move bestMove = orderedMoves[0];
            int bestScore = 0;
            int bestDepth = 0;
            List<Move> bestPv = new() { bestMove };
            bool reachedMaxDepth = false;
            int seedIndex = 0;

            int Remaining() => limits.MoveTimeMs - (int)Math.Min(stopwatch.ElapsedMilliseconds, int.MaxValue - 1);

            // 判定"我方此着直接将死/困毙对方":子局面无合法着法,任何深度下都成立。
            static bool IsImmediateMate(SearchResult childResult) =>
                childResult.Completed && childResult.Depth == 0 && childResult.PrincipalVariation.Count == 0;

            // 根层同步迭代加深:一轮内全部根着法并行、全窗口算到同一子树深度,
            // 整轮算完才采纳比较结果;没算完的轮次整轮作废,沿用上一轮结论,
            // 杜绝拿浅层乐观评分压过深层真实评分的旧病。上一轮最佳着法排在
            // 本轮首位。实测注:曾试过共享 α 收窄子树窗口(PVS/零窗试探两种),
            // 均因本引擎 fail-hard 截断证伪昂贵+跨轮 TT 边界污染而净变慢,勿回退。
            for (int depth = 2; depth <= limits.MaxDepth; depth++)
            {
                int childDepth = depth - 1;
                bool roundAborted = false;
                int roundBestScore = int.MinValue + 1;
                int roundBestIndex = -1;
                List<Move>? roundBestPv = null;

                var roundOrder = new List<int>(orderedMoves.Count) { seedIndex };
                for (int i = 0; i < orderedMoves.Count; i++)
                {
                    if (i != seedIndex)
                        roundOrder.Add(i);
                }

                try
                {
                    // 故意不给 ParallelOptions 传 CancellationToken:协作取消会让
                    // ForEach 在超时瞬间抛 OCE 并等待全部副本任务汇合,该汇合在
                    // net10 上出现过挂死(2026-08-17 对局实测,线程卡在 EH.DispatchEx)。
                    // 取消一律由委托入口自查 + 子树搜索内部 ShouldStop 快速退出。
                    Parallel.ForEach(
                        roundOrder,
                        new ParallelOptions
                        {
                            MaxDegreeOfParallelism = Math.Max(1, _options.RootParallelism)
                        },
                        i =>
                        {
                            if (Volatile.Read(ref roundAborted) || token.IsCancellationRequested)
                            {
                                Volatile.Write(ref roundAborted, true);
                                return;
                            }

                            int remaining = Remaining();
                            if (remaining <= 0)
                            {
                                Volatile.Write(ref roundAborted, true);
                                return;
                            }

                            int a = Math.Max(Volatile.Read(ref roundBestScore), -_options.MateScore);
                            var childLimits = new SearchLimits(childDepth, remaining, limits.QuiescenceDepth);
                            var childResult = workers[i].SearchSubtree(childBoards[i], childLimits, null, null, token);
                            lock (sync)
                            {
                                totalNodes += childResult.Nodes;
                            }

                            int score;
                            List<Move> pv;
                            if (IsImmediateMate(childResult))
                            {
                                score = -childResult.Score;
                                pv = new List<Move> { orderedMoves[i] };
                            }
                            else
                            {
                                if (!childResult.Completed || childResult.Depth < childDepth)
                                {
                                    Volatile.Write(ref roundAborted, true);
                                    return;
                                }

                                score = -childResult.Score;
                                if (score <= a)
                                    return; // 确证不优于已知最佳,淘汰。

                                pv = new List<Move> { orderedMoves[i] };
                                pv.AddRange(childResult.PrincipalVariation);
                            }

                            lock (sync)
                            {
                                if (score > roundBestScore)
                                {
                                    roundBestScore = score;
                                    roundBestIndex = i;
                                    roundBestPv = pv;
                                }
                            }
                        });
                }
                catch (Exception)
                {
                    // 任何并行执行期异常(含聚合异常)都只作废本轮,沿用上一轮结论,
                    // 绝不让整个搜索调用挂掉或悬死。
                    roundAborted = true;
                }

                if (roundAborted || roundBestIndex < 0)
                    break;

                bestMove = orderedMoves[roundBestIndex];
                bestScore = roundBestScore;
                bestDepth = depth;
                bestPv = roundBestPv ?? new List<Move> { bestMove };
                seedIndex = roundBestIndex;

                if (depth == limits.MaxDepth)
                    reachedMaxDepth = true;

                // 已找到必胜杀着,继续加深无意义。
                if (bestScore >= _options.MateScore - 1000)
                    break;
            }

            stopwatch.Stop();
            bool completed = !cancellationToken.IsCancellationRequested
                && (reachedMaxDepth || bestScore >= _options.MateScore - 1000);
            return new SearchResult(bestMove, bestScore, Math.Max(1, bestDepth), totalNodes, stopwatch.Elapsed, bestPv, completed);
        }

        internal SearchResult SearchSubtree(Board board, SearchLimits limits, int? alpha, int? beta, CancellationToken cancellationToken)
        {
            return _search.Search(board, limits, cancellationToken, alpha, beta);
        }

        private TraditionalEngine CreateWorkerWithoutBookMove()
        {
            return new TraditionalEngine(new TraditionalEngineOptions
            {
                MateScore = _options.MateScore,
                UseQuiescenceSearch = _options.UseQuiescenceSearch,
                SkipPerpetualCheckInsideSearch = _options.SkipPerpetualCheckInsideSearch,
                SkipPerpetualCheckAtRoot = _options.SkipPerpetualCheckAtRoot,
                TranspositionTableEntries = Math.Max(16_384, _options.TranspositionTableEntries / Math.Max(1, _options.RootParallelism)),
                MateSearchPly = _options.MateSearchPly,
                UseNullMovePruning = _options.UseNullMovePruning,
                UseFutilityPruning = _options.UseFutilityPruning,
                UseRazoring = _options.UseRazoring,
                UseSeePruning = _options.UseSeePruning,
                OpeningBook = null,
                OpeningBookMode = OpeningBookMode.Off,
                MoveOrderingBook = _options.MoveOrderingBook,
                MasterKnowledgeBook = _options.MasterKnowledgeBook,
                RootParallelism = 1
            });
        }
    }
}
