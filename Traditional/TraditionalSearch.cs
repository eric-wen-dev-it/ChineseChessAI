using System.Diagnostics;
using ChineseChessAI.Core;

namespace ChineseChessAI.Traditional
{
    public sealed class TraditionalSearch
    {
        private readonly MoveGenerator _generator;
        private readonly TraditionalEvaluator _evaluator;
        private readonly TraditionalMoveOrdering _moveOrdering;
        private readonly TraditionalEngineOptions _options;
        private readonly TranspositionTable _table;

        private SearchLimits _limits;
        private Stopwatch _stopwatch = new();
        private long _nodes;
        private Move _bestMove;
        private int _completedDepth;
        private bool _stopRequested;
        private List<Move> _principalVariation = new();
        private readonly Move?[] _killerOne = new Move?[128];
        private readonly Move?[] _killerTwo = new Move?[128];
        private readonly int[] _history = new int[8100];
        // "对方有立即杀威胁"按局面哈希记忆化:静搜叶子高度重复,该检测又要
        // 克隆棋盘+全量着法生成+逐着法找杀,是静搜每节点成本的大头之一。
        // 纯加速,不改变任何着法选择。
        private readonly Dictionary<ulong, bool> _mateThreatCache = new();

        public TraditionalSearch(
            MoveGenerator generator,
            TraditionalEvaluator evaluator,
            TraditionalMoveOrdering moveOrdering,
            TraditionalEngineOptions options,
            TranspositionTable table)
        {
            _generator = generator;
            _evaluator = evaluator;
            _moveOrdering = moveOrdering;
            _options = options;
            _table = table;
        }

        public SearchResult Search(Board board, SearchLimits limits, CancellationToken cancellationToken = default)
        {
            return Search(board, limits, cancellationToken, null, null);
        }

        // startDepth:迭代加深的起始深度(Lazy SMP 辅助线程错位用,主线程恒 1)。
        // 大于 1 时跳过浅层迭代,直接从该深度起搜——首轮无浅层最佳着法播种,但在
        // 共享置换表已被其它线程/上一手预热时反而更快抵达深层,给主线程回灌深条目。
        public SearchResult Search(Board board, SearchLimits limits, CancellationToken cancellationToken, int? externalAlpha, int? externalBeta, int startDepth = 1)
        {
            _limits = limits;
            _nodes = 0;
            _completedDepth = 0;
            _stopRequested = false;
            _principalVariation = new List<Move>();
            Array.Clear(_killerOne);
            Array.Clear(_killerTwo);
            Array.Clear(_history);
            _mateThreatCache.Clear();
            _stopwatch = Stopwatch.StartNew();

            try
            {
                var rootMoves = _generator.GenerateLegalMoves(board, skipPerpetualCheck: _options.SkipPerpetualCheckAtRoot);
                if (rootMoves.Count == 0)
                {
                    return new SearchResult(default, -_options.MateScore, 0, 0, _stopwatch.Elapsed, Array.Empty<Move>(), true);
                }

                _bestMove = rootMoves[0];
                int bestScore = int.MinValue + 1;
                // 外部边界(根并行的共享 α 窗口)是硬约束;内部吸入窗口在其内收窄,
                // 吸入失败最多放宽回外部边界,绝不越界。无外部边界时即 ±MateScore,
                // 行为与原版一致。
                int floorAlpha = externalAlpha ?? -_options.MateScore;
                int ceilBeta = externalBeta ?? _options.MateScore;
                int firstDepth = Math.Clamp(startDepth, 1, Math.Max(1, limits.MaxDepth));
                for (int depth = firstDepth; depth <= Math.Max(1, limits.MaxDepth); depth++)
                {
                    if (ShouldStop(cancellationToken))
                        break;

                    bool useAspiration = depth >= 4 && bestScore > int.MinValue / 2;
                    int alpha = useAspiration ? Math.Max(floorAlpha, bestScore - 80) : floorAlpha;
                    int beta = useAspiration ? Math.Min(ceilBeta, bestScore + 80) : ceilBeta;
                    if (alpha >= beta)
                    {
                        alpha = floorAlpha;
                        beta = ceilBeta;
                    }
                    int windowAlpha = alpha;
                    int windowBeta = beta;
                    Move depthBestMove = _bestMove;
                    int depthBestScore = int.MinValue + 1;
                    List<Move> depthBestPv = new();
                    bool retryFullWindow = false;

                RetryRoot:
                    foreach (var move in _moveOrdering.OrderMoves(board, rootMoves, _bestMove, null, null, _history))
                    {
                        if (ShouldStop(cancellationToken))
                            break;
                        board.Push(move.From, move.To);
                        try
                        {
                            int score = -Negamax(board, depth - 1, -beta, -alpha, 1, 2, true, out var childPv, cancellationToken);
                            if (_stopRequested)
                                break;

                            if (score > depthBestScore)
                            {
                                depthBestScore = score;
                                depthBestMove = move;
                                depthBestPv = new List<Move> { move };
                                depthBestPv.AddRange(childPv);
                            }

                            if (score > alpha)
                                alpha = score;
                        }
                        finally
                        {
                            board.Pop();
                        }
                    }

                    if (_stopRequested)
                        break;

                    if (!retryFullWindow
                        && (depthBestScore <= windowAlpha || depthBestScore >= windowBeta)
                        && (windowAlpha != floorAlpha || windowBeta != ceilBeta))
                    {
                        alpha = floorAlpha;
                        beta = ceilBeta;
                        windowAlpha = alpha;
                        windowBeta = beta;
                        depthBestScore = int.MinValue + 1;
                        depthBestPv.Clear();
                        retryFullWindow = true;
                        goto RetryRoot;
                    }

                    _bestMove = depthBestMove;
                    bestScore = depthBestScore;
                    _completedDepth = depth;
                    _principalVariation = depthBestPv;
                }

                _stopwatch.Stop();
                return new SearchResult(_bestMove, bestScore, _completedDepth, _nodes, _stopwatch.Elapsed, _principalVariation, !_stopRequested);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                _stopRequested = true;
                _stopwatch.Stop();
                return new SearchResult(_bestMove, _evaluator.Evaluate(board), _completedDepth, _nodes, _stopwatch.Elapsed, _principalVariation, false);
            }
        }

        private int Negamax(Board board, int depth, int alpha, int beta, int ply, int checkExtensionsLeft, bool allowNullMove, out List<Move> principalVariation, CancellationToken cancellationToken)
        {
            principalVariation = new List<Move>();
            if (ShouldStop(cancellationToken))
                return _evaluator.Evaluate(board);
            _nodes++;
            int originalAlpha = alpha;
            Move? ttMove = null;

            if (board.GetRepetitionCount() >= 3)
                return 0;

            if (_table.TryGet(board.CurrentHash, ply, _options.MateScore, out var entry) && entry.Depth >= depth)
            {
                ttMove = entry.BestMove;
                if (entry.Bound == TTBound.Exact)
                    return entry.Score;
                if (entry.Bound == TTBound.Lower)
                    alpha = Math.Max(alpha, entry.Score);
                else if (entry.Bound == TTBound.Upper)
                    beta = Math.Min(beta, entry.Score);

                if (alpha >= beta)
                    return entry.Score;
            }

            if (depth <= 0)
            {
                return _options.UseQuiescenceSearch
                    ? Quiescence(board, _limits.QuiescenceDepth, alpha, beta, ply, 0, cancellationToken)
                    : _evaluator.Evaluate(board);
            }

            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
            if (!inCheck && _options.UseRazoring && depth == 1)
            {
                int staticScore = _evaluator.Evaluate(board);
                if (staticScore + 180 <= alpha)
                    return Quiescence(board, _limits.QuiescenceDepth / 2, alpha, beta, ply, 0, cancellationToken);
            }

            if (!inCheck && _options.UseNullMovePruning && allowNullMove && depth >= 3 && HasNonPawnMaterial(board, board.IsRedTurn))
            {
                var nullBoard = board.Clone();
                nullBoard.SwitchTurnPreservingHistory();
                int reduction = depth >= 5 ? 3 : 2;
                int nullScore = -Negamax(nullBoard, depth - 1 - reduction, -beta, -beta + 1, ply + 1, checkExtensionsLeft, false, out _, cancellationToken);
                if (!_stopRequested && nullScore >= beta)
                    return beta;
            }

            var moves = _generator.GenerateLegalMoves(
                board,
                skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);

            if (moves.Count == 0)
                return -_options.MateScore + ply;

            if (inCheck && checkExtensionsLeft > 0)
            {
                depth++;
                checkExtensionsLeft--;
            }

            // 一遍扫描同时完成:立杀检测(将军且对方无应着)+给强制杀搜索备好
            // "哪些着法是将军"的标记,替代旧版的两遍逐着法 Push+IsChecking 加一次
            // 重复着法生成。
            bool[] givesCheckFlags = new bool[moves.Count];
            bool anyCheckingMove = false;
            for (int i = 0; i < moves.Count; i++)
            {
                var move = moves[i];
                board.Push(move.From, move.To);
                try
                {
                    if (!_generator.IsChecking(board, !board.IsRedTurn))
                        continue;

                    givesCheckFlags[i] = true;
                    anyCheckingMove = true;
                    var replies = _generator.GenerateLegalMoves(
                        board,
                        skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);
                    if (replies.Count == 0)
                    {
                        principalVariation.Add(move);
                        _table.Store(board.CurrentHash, depth, _options.MateScore - ply, move, TTBound.Exact, ply, _options.MateScore);
                        return _options.MateScore - ply;
                    }
                }
                finally
                {
                    board.Pop();
                }
            }

            if (_options.MateSearchPly >= 3 && anyCheckingMove
                && TryFindForcedCheckmateFromFlags(board, moves, givesCheckFlags, Math.Min(_options.MateSearchPly, depth + 2), out var forcedMateMove, cancellationToken))
            {
                principalVariation.Add(forcedMateMove);
                _table.Store(board.CurrentHash, depth, _options.MateScore - ply - 2, forcedMateMove, TTBound.Exact, ply, _options.MateScore);
                return _options.MateScore - ply - 2;
            }

            Move bestMove = moves[0];
            List<Move> bestChildPv = new();
            int bestScore = int.MinValue + 1;
            int moveIndex = 0;
            bool foundPv = false;
            Move? killerOne = ply < _killerOne.Length ? _killerOne[ply] : null;
            Move? killerTwo = ply < _killerTwo.Length ? _killerTwo[ply] : null;

            foreach (var move in _moveOrdering.OrderMoves(board, moves, ttMove, killerOne, killerTwo, _history))
            {
                bool isCapture = board.GetPiece(move.To) != 0;
                if (!isCapture && board.WillCauseThreefoldRepetition(move.From, move.To))
                {
                    int drawScore = 0;
                    if (drawScore > bestScore)
                    {
                        bestScore = drawScore;
                        bestMove = move;
                    }

                    if (drawScore > alpha)
                    {
                        alpha = drawScore;
                        bestMove = move;
                        bestChildPv = new List<Move>();
                        foundPv = true;
                        if (alpha >= beta)
                            break;
                    }

                    moveIndex++;
                    continue;
                }

                int staticEval = 0;
                if (!inCheck && _options.UseFutilityPruning && depth <= 2 && !isCapture && moveIndex >= 4)
                {
                    staticEval = staticEval == 0 ? _evaluator.Evaluate(board) : staticEval;
                    if (staticEval + 180 * depth <= alpha)
                    {
                        moveIndex++;
                        continue;
                    }
                }

                if (_options.UseSeePruning && isCapture && depth <= 3 && IsPotentiallyBadCapture(board, move) && EstimateSee(board, move) < -120)
                {
                    moveIndex++;
                    continue;
                }

                int reduction = 0;
                if (!inCheck && !isCapture && depth >= 4 && moveIndex >= 4)
                    reduction = moveIndex >= 10 && depth >= 5 ? 2 : 1;

                board.Push(move.From, move.To);
                try
                {
                    bool givesCheck = _generator.IsChecking(board, !board.IsRedTurn);
                    int extension = givesCheck && checkExtensionsLeft > 0 ? 1 : 0;
                    int nextExtensionsLeft = checkExtensionsLeft - extension;
                    int nextDepth = Math.Max(0, depth - 1 + extension - reduction);
                    int score;
                    List<Move> childPv;
                    if (foundPv)
                    {
                        score = -Negamax(board, nextDepth, -alpha - 1, -alpha, ply + 1, nextExtensionsLeft, true, out childPv, cancellationToken);
                        if (!_stopRequested && score > alpha && score < beta)
                            score = -Negamax(board, Math.Max(0, depth - 1 + extension), -beta, -alpha, ply + 1, nextExtensionsLeft, true, out childPv, cancellationToken);
                    }
                    else
                    {
                        score = -Negamax(board, Math.Max(0, depth - 1 + extension), -beta, -alpha, ply + 1, nextExtensionsLeft, true, out childPv, cancellationToken);
                    }

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestMove = move;
                    }

                    if (score > alpha)
                    {
                        alpha = score;
                        bestMove = move;
                        bestChildPv = childPv;
                        foundPv = true;
                        if (alpha >= beta)
                        {
                            if (!isCapture)
                            {
                                StoreKiller(ply, move);
                                _history[move.ToNetworkIndex()] += depth * depth;
                            }
                            break;
                        }
                    }
                }
                finally
                {
                    board.Pop();
                }

                moveIndex++;
            }

            TTBound bound = bestScore <= originalAlpha ? TTBound.Upper : (bestScore >= beta ? TTBound.Lower : TTBound.Exact);
            _table.Store(board.CurrentHash, depth, bestScore, bestMove, bound, ply, _options.MateScore);

            principalVariation.Add(bestMove);
            principalVariation.AddRange(bestChildPv);
            return bestScore;
        }

        private int Quiescence(Board board, int depth, int alpha, int beta, int ply, int qPly, CancellationToken cancellationToken)
        {
            if (ShouldStop(cancellationToken))
                return _evaluator.Evaluate(board);
            _nodes++;

            // 静搜置换表:任何既有条目(静搜深度 0 或主搜索更深)对静搜节点都够用。
            // 本节点哈希必须在此处捕获:着法循环里 Push 之后 board.CurrentHash 是
            // 子局面的,拿它存分会张冠李戴毒化置换表(首版实弹教训,2:6 惨案)。
            ulong nodeHash = board.CurrentHash;
            Move? ttMove = null;
            if (_options.UseQuiescenceTT && _table.TryGet(nodeHash, ply, _options.MateScore, out var qEntry))
            {
                ttMove = qEntry.BestMove;
                if (qEntry.Bound == TTBound.Exact)
                    return qEntry.Score;
                if (qEntry.Bound == TTBound.Lower)
                    alpha = Math.Max(alpha, qEntry.Score);
                else if (qEntry.Bound == TTBound.Upper)
                    beta = Math.Min(beta, qEntry.Score);

                if (alpha >= beta)
                    return qEntry.Score;
            }

            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
            var legalMoves = _generator.GenerateLegalMoves(
                board,
                skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);

            if (legalMoves.Count == 0)
                return -_options.MateScore + ply;

            if (inCheck)
            {
                if (depth <= 0)
                    return _evaluator.Evaluate(board);

                foreach (var move in _moveOrdering.OrderMoves(board, legalMoves, ttMove))
                {
                    board.Push(move.From, move.To);
                    try
                    {
                        int score = -Quiescence(board, Math.Max(0, depth - 1), -beta, -alpha, ply + 1, qPly + 1, cancellationToken);
                        if (score >= beta)
                            return beta;
                        if (score > alpha)
                            alpha = score;
                    }
                    finally
                    {
                        board.Pop();
                    }
                }

                return alpha;
            }

            // 战术着法筛选。前 QuiescenceCheckPlies 层沿用一遍扫描(吃子或将军
            // 都算战术着法,顺带立杀检测);更深层退化为纯吃子延伸——省掉对全部
            // 合法着法逐一 Push+IsChecking 的探测,这是静搜每节点成本的大头。
            bool includeChecks = qPly < _options.QuiescenceCheckPlies;
            var tacticalMoves = new List<Move>();
            if (includeChecks)
            {
                foreach (var move in legalMoves)
                {
                    bool isCapture = board.GetPiece(move.To) != 0;
                    board.Push(move.From, move.To);
                    try
                    {
                        if (_generator.IsChecking(board, !board.IsRedTurn))
                        {
                            var replies = _generator.GenerateLegalMoves(
                                board,
                                skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);
                            if (replies.Count == 0)
                                return _options.MateScore - ply - 1;

                            tacticalMoves.Add(move);
                        }
                        else if (isCapture)
                        {
                            tacticalMoves.Add(move);
                        }
                    }
                    finally
                    {
                        board.Pop();
                    }
                }
            }
            else
            {
                foreach (var move in legalMoves)
                {
                    if (board.GetPiece(move.To) != 0)
                        tacticalMoves.Add(move);
                }
            }

            // 对方存在"立即将死"威胁时,旧逻辑直接返回我方被杀的杀分——但安静
            // 防着(如退士解杀)在静态搜索里不可见,威胁≠必死,这是实战假杀根因
            // (2026-08-17 俥8进9 事故,根局面可复现)。改为:禁止 stand-pat,像
            // 应将一样遍历全部合法着法找解;静搜预算耗尽无法核实时返回重罚分
            // (量级远超子力但不进杀分区间,不触发"找到必杀"早停)。
            if (OpponentHasImmediateMateAtLeafCached(board, cancellationToken))
            {
                if (depth <= 0)
                    return -_options.MateScore / 4;

                foreach (var move in _moveOrdering.OrderMoves(board, legalMoves, ttMove))
                {
                    board.Push(move.From, move.To);
                    try
                    {
                        int score = -Quiescence(board, depth - 1, -beta, -alpha, ply + 1, qPly + 1, cancellationToken);
                        if (score >= beta)
                            return beta;
                        if (score > alpha)
                            alpha = score;
                    }
                    finally
                    {
                        board.Pop();
                    }
                }

                return alpha;
            }

            int standPat = _evaluator.Evaluate(board);
            if (standPat >= beta)
            {
                if (_options.UseQuiescenceTT)
                    _table.StoreQuiescence(nodeHash, beta, ttMove ?? default, TTBound.Lower, ply, _options.MateScore);
                return beta;
            }

            if (standPat > alpha)
                alpha = standPat;
            if (depth <= 0)
                return alpha;

            // 以下 alpha 已含 stand-pat 抬升;searchAlpha 用于收尾时判定 Upper/Exact。
            int searchAlpha = alpha;
            Move bestTacticalMove = default;
            foreach (var move in _moveOrdering.OrderMoves(board, tacticalMoves, ttMove))
            {
                sbyte victim = board.GetPiece(move.To);
                if (victim != 0)
                {
                    // Delta 剪枝:吃到这个子加上边际都追不上 alpha,搜了也白搜。
                    if (_options.UseQuiescenceDeltaPruning
                        && standPat + PieceValue(victim) + _options.QuiescenceDeltaMargin <= alpha)
                        continue;

                    // 静态交换明亏的吃子(小子换不回来)直接跳过。
                    if (_options.UseQuiescenceSeePruning
                        && IsPotentiallyBadCapture(board, move)
                        && EstimateSee(board, move) < 0)
                        continue;
                }

                board.Push(move.From, move.To);
                try
                {
                    int score = -Quiescence(board, depth - 1, -beta, -alpha, ply + 1, qPly + 1, cancellationToken);
                    if (score >= beta)
                    {
                        if (_options.UseQuiescenceTT && !_stopRequested)
                            _table.StoreQuiescence(nodeHash, beta, move, TTBound.Lower, ply, _options.MateScore);
                        return beta;
                    }

                    if (score > alpha)
                    {
                        alpha = score;
                        bestTacticalMove = move;
                    }
                }
                finally
                {
                    board.Pop();
                }
            }

            if (_options.UseQuiescenceTT && !_stopRequested)
            {
                TTBound bound = alpha > searchAlpha ? TTBound.Exact : TTBound.Upper;
                _table.StoreQuiescence(nodeHash, alpha, bestTacticalMove, bound, ply, _options.MateScore);
            }

            return alpha;
        }

        private bool OpponentHasImmediateMateAtLeafCached(Board board, CancellationToken cancellationToken)
        {
            if (_mateThreatCache.TryGetValue(board.CurrentHash, out bool cached))
                return cached;

            bool result = OpponentHasImmediateMateAtLeaf(board, cancellationToken);
            // 停表/取消途中算出的结果不可信,不入缓存(整层结果本来也会作废)。
            if (!_stopRequested)
                _mateThreatCache[board.CurrentHash] = result;
            return result;
        }

        private bool TryFindImmediateMate(Board board, List<Move> legalMoves, out Move mateMove, CancellationToken cancellationToken)
        {
            mateMove = default;
            // 存在性检测,与着法顺序无关,不排序。
            foreach (var move in legalMoves)
            {
                if (ShouldStop(cancellationToken))
                    return false;
                board.Push(move.From, move.To);
                try
                {
                    if (!_generator.IsChecking(board, !board.IsRedTurn))
                        continue;

                    var replies = _generator.GenerateLegalMoves(
                        board,
                        skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);
                    if (replies.Count == 0)
                    {
                        mateMove = move;
                        return true;
                    }
                }
                finally
                {
                    board.Pop();
                }
            }

            return false;
        }

        private bool OpponentHasImmediateMateAtLeaf(Board board, CancellationToken cancellationToken)
        {
            if (ShouldStop(cancellationToken))
                return false;
            var opponentBoard = board.Clone();
            opponentBoard.SwitchTurnPreservingHistory();
            if (!_generator.IsKingSafe(opponentBoard, opponentBoard.IsRedTurn))
                return false;

            var opponentMoves = _generator.GenerateLegalMoves(
                opponentBoard,
                skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);
            return TryFindImmediateMate(opponentBoard, opponentMoves, out _, cancellationToken);
        }

        private bool GivesCheck(Board board, Move move)
        {
            board.Push(move.From, move.To);
            try
            {
                return _generator.IsChecking(board, !board.IsRedTurn);
            }
            finally
            {
                board.Pop();
            }
        }

        // 顶层入口:调用方已算好哪些着法是将军(givesCheckFlags),只沿将军着法递归。
        private bool TryFindForcedCheckmateFromFlags(Board board, List<Move> moves, bool[] givesCheckFlags, int remainingPly, out Move mateMove, CancellationToken cancellationToken)
        {
            mateMove = default;
            if (remainingPly <= 0 || ShouldStop(cancellationToken))
                return false;

            for (int i = 0; i < moves.Count; i++)
            {
                if (!givesCheckFlags[i])
                    continue;

                var move = moves[i];
                board.Push(move.From, move.To);
                try
                {
                    var replies = _generator.GenerateLegalMoves(
                        board,
                        skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);
                    if (replies.Count == 0)
                    {
                        mateMove = move;
                        return true;
                    }

                    if (remainingPly <= 1)
                        continue;

                    bool allRepliesLose = true;
                    foreach (var reply in replies)
                    {
                        board.Push(reply.From, reply.To);
                        try
                        {
                            if (!TryFindForcedCheckmate(board, remainingPly - 2, out _, cancellationToken))
                            {
                                allRepliesLose = false;
                                break;
                            }
                        }
                        finally
                        {
                            board.Pop();
                        }
                    }

                    if (allRepliesLose)
                    {
                        mateMove = move;
                        return true;
                    }
                }
                finally
                {
                    board.Pop();
                }
            }

            return false;
        }

        private bool TryFindForcedCheckmate(Board board, int remainingPly, out Move mateMove, CancellationToken cancellationToken)
        {
            mateMove = default;
            if (remainingPly <= 0 || ShouldStop(cancellationToken))
                return false;

            var legalMoves = _generator.GenerateLegalMoves(
                board,
                skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);

            foreach (var move in legalMoves)
            {
                if (board.GetPiece(move.To) == 0 && !GivesCheck(board, move))
                    continue;

                board.Push(move.From, move.To);
                try
                {
                    if (!_generator.IsChecking(board, !board.IsRedTurn))
                        continue;

                    var replies = _generator.GenerateLegalMoves(
                        board,
                        skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);
                    if (replies.Count == 0)
                    {
                        mateMove = move;
                        return true;
                    }

                    if (remainingPly <= 1)
                        continue;

                    bool allRepliesLose = true;
                    foreach (var reply in replies)
                    {
                        board.Push(reply.From, reply.To);
                        try
                        {
                            if (!TryFindForcedCheckmate(board, remainingPly - 2, out _, cancellationToken))
                            {
                                allRepliesLose = false;
                                break;
                            }
                        }
                        finally
                        {
                            board.Pop();
                        }
                    }

                    if (allRepliesLose)
                    {
                        mateMove = move;
                        return true;
                    }
                }
                finally
                {
                    board.Pop();
                }
            }

            return false;
        }

        private static bool HasNonPawnMaterial(Board board, bool red)
        {
            for (int i = 0; i < 90; i++)
            {
                sbyte piece = board.GetPiece(i);
                if (piece == 0 || (piece > 0) != red)
                    continue;

                int type = Math.Abs(piece);
                if (type is 4 or 5 or 6)
                    return true;
            }

            return false;
        }

        private int EstimateSee(Board board, Move move)
        {
            return StaticExchangeEvaluator.Evaluate(board, move, _generator);
        }

        private static bool IsPotentiallyBadCapture(Board board, Move move)
        {
            sbyte attacker = board.GetPiece(move.From);
            sbyte victim = board.GetPiece(move.To);
            if (attacker == 0 || victim == 0)
                return false;

            return PieceValue(victim) <= PieceValue(attacker);
        }

        private static int PieceValue(sbyte piece)
        {
            return Math.Abs(piece) switch
            {
                1 => 10_000,
                2 => 200,
                3 => 200,
                4 => 450,
                5 => 900,
                6 => 400,
                7 => 100,
                _ => 0
            };
        }

        private void StoreKiller(int ply, Move move)
        {
            if (ply >= _killerOne.Length)
                return;

            if (_killerOne[ply].HasValue && _killerOne[ply]!.Value.Equals(move))
                return;

            _killerTwo[ply] = _killerOne[ply];
            _killerOne[ply] = move;
        }

        private bool ShouldStop(CancellationToken cancellationToken)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                _stopRequested = true;
                return true;
            }

            if (_limits.MoveTimeMs > 0 && _stopwatch.ElapsedMilliseconds >= _limits.MoveTimeMs)
            {
                _stopRequested = true;
                return true;
            }

            return false;
        }
    }
}
