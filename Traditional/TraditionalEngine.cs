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
            _moveOrdering = new TraditionalMoveOrdering(_generator, _options.MoveOrderingBook ?? _options.OpeningBook);
            var table = new TranspositionTable(_options.TranspositionTableEntries);
            _search = new TraditionalSearch(_generator, evaluator, _moveOrdering, _options, table);
        }

        public SearchResult Search(Board board, SearchLimits limits, CancellationToken cancellationToken = default)
        {
            if (_options.OpeningBook != null && _options.OpeningBook.TryGetMove(board, _options.OpeningBookMode, out var bookMove))
            {
                return new SearchResult(bookMove, 0, 0, 0, TimeSpan.Zero, new[] { bookMove }, true);
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
            var rootMoves = _generator.GenerateLegalMoves(board, skipPerpetualCheck: false, cancellationToken);
            if (rootMoves.Count == 0)
                return new SearchResult(default, -_options.MateScore, 0, 0, stopwatch.Elapsed, Array.Empty<Move>(), true);

            var orderedMoves = _moveOrdering.OrderMoves(board, rootMoves);
            using var timeCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeCts.CancelAfter(limits.MoveTimeMs);
            var token = timeCts.Token;

            object sync = new();
            Move bestMove = orderedMoves[0];
            int bestScore = int.MinValue + 1;
            int bestDepth = 0;
            long totalNodes = 0;
            int bestOrder = int.MaxValue;
            bool allCompleted = true;
            List<Move> bestPv = new() { bestMove };

            try
            {
                Parallel.ForEach(
                    orderedMoves.Select((move, index) => (Move: move, Order: index)),
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = Math.Max(1, _options.RootParallelism),
                        CancellationToken = token
                    },
                    item =>
                    {
                        token.ThrowIfCancellationRequested();

                        var childBoard = board.Clone();
                        childBoard.Push(item.Move.From, item.Move.To);

                        var worker = CreateWorkerWithoutBookMove();
                        int childDepth = limits.MoveTimeMs < 10_000
                            ? Math.Max(1, limits.MaxDepth - 2)
                            : Math.Max(1, limits.MaxDepth - 1);
                        var childLimits = new SearchLimits(
                            childDepth,
                            Math.Max(1, limits.MoveTimeMs - (int)Math.Min(stopwatch.ElapsedMilliseconds, int.MaxValue - 1)),
                            limits.QuiescenceDepth);

                        var childResult = worker.Search(childBoard, childLimits, token);
                        int score;
                        int depth;
                        if (childResult.Depth <= 0 || Math.Abs(childResult.Score) > _options.MateScore)
                        {
                            score = -new TraditionalEvaluator().Evaluate(childBoard);
                            depth = 1;
                        }
                        else
                        {
                            score = -childResult.Score;
                            depth = childResult.Depth + 1;
                        }
                        var pv = new List<Move> { item.Move };
                        pv.AddRange(childResult.PrincipalVariation);

                        lock (sync)
                        {
                            totalNodes += childResult.Nodes;
                            allCompleted &= childResult.Completed;
                            if (score > bestScore || (score == bestScore && depth > bestDepth) || (score == bestScore && depth == bestDepth && item.Order < bestOrder))
                            {
                                bestScore = score;
                                bestDepth = depth;
                                bestMove = item.Move;
                                bestOrder = item.Order;
                                bestPv = pv;
                            }
                        }
                    });
            }
            catch (OperationCanceledException)
            {
                allCompleted = false;
            }

            stopwatch.Stop();
            bool completed = allCompleted && !cancellationToken.IsCancellationRequested && stopwatch.ElapsedMilliseconds < limits.MoveTimeMs;
            return new SearchResult(bestMove, bestScore, bestDepth, totalNodes, stopwatch.Elapsed, bestPv, completed);
        }

        private TraditionalEngine CreateWorkerWithoutBookMove()
        {
            return new TraditionalEngine(new TraditionalEngineOptions
            {
                MateScore = _options.MateScore,
                UseQuiescenceSearch = _options.UseQuiescenceSearch,
                SkipPerpetualCheckInsideSearch = _options.SkipPerpetualCheckInsideSearch,
                TranspositionTableEntries = Math.Max(16_384, _options.TranspositionTableEntries / Math.Max(1, _options.RootParallelism)),
                MateSearchPly = _options.MateSearchPly,
                UseNullMovePruning = _options.UseNullMovePruning,
                UseFutilityPruning = _options.UseFutilityPruning,
                UseRazoring = _options.UseRazoring,
                UseSeePruning = _options.UseSeePruning,
                OpeningBook = null,
                OpeningBookMode = OpeningBookMode.Off,
                MoveOrderingBook = _options.MoveOrderingBook,
                RootParallelism = 1
            });
        }
    }
}
