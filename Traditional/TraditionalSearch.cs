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
            _limits = limits;
            _nodes = 0;
            _completedDepth = 0;
            _stopRequested = false;
            _principalVariation = new List<Move>();
            Array.Clear(_killerOne);
            Array.Clear(_killerTwo);
            Array.Clear(_history);
            _stopwatch = Stopwatch.StartNew();

            try
            {
                var rootMoves = _generator.GenerateLegalMoves(board, skipPerpetualCheck: false);
                if (rootMoves.Count == 0)
                {
                    return new SearchResult(default, -_options.MateScore, 0, 0, _stopwatch.Elapsed, Array.Empty<Move>(), true);
                }

                _bestMove = rootMoves[0];
                int bestScore = int.MinValue + 1;
                for (int depth = 1; depth <= Math.Max(1, limits.MaxDepth); depth++)
                {
                    if (ShouldStop(cancellationToken))
                        break;

                    int alpha = depth >= 4 && bestScore > int.MinValue / 2 ? bestScore - 80 : -_options.MateScore;
                    int beta = depth >= 4 && bestScore > int.MinValue / 2 ? bestScore + 80 : _options.MateScore;
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

                    if (!retryFullWindow && (depthBestScore <= windowAlpha || depthBestScore >= windowBeta))
                    {
                        alpha = -_options.MateScore;
                        beta = _options.MateScore;
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

            if (_table.TryGet(board.CurrentHash, out var entry) && entry.Depth >= depth)
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
                    ? Quiescence(board, _limits.QuiescenceDepth, alpha, beta, cancellationToken)
                    : _evaluator.Evaluate(board);
            }

            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
            if (!inCheck && _options.UseRazoring && depth == 1)
            {
                int staticScore = _evaluator.Evaluate(board);
                if (staticScore + 180 <= alpha)
                    return Quiescence(board, _limits.QuiescenceDepth / 2, alpha, beta, cancellationToken);
            }

            if (!inCheck && _options.UseNullMovePruning && allowNullMove && depth >= 3 && HasNonPawnMaterial(board, board.IsRedTurn))
            {
                var nullBoard = board.Clone();
                nullBoard.LoadState(board.GetState(), !board.IsRedTurn);
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

            if (TryFindImmediateMate(board, moves, out var mateMove, cancellationToken))
            {
                principalVariation.Add(mateMove);
                _table.Store(board.CurrentHash, depth, _options.MateScore - ply, mateMove, TTBound.Exact);
                return _options.MateScore - ply;
            }

            if (_options.MateSearchPly >= 3 && TryFindForcedCheckmate(board, Math.Min(_options.MateSearchPly, depth + 2), out var forcedMateMove, cancellationToken))
            {
                principalVariation.Add(forcedMateMove);
                _table.Store(board.CurrentHash, depth, _options.MateScore - ply - 2, forcedMateMove, TTBound.Exact);
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

            TTBound bound = alpha <= originalAlpha ? TTBound.Upper : (alpha >= beta ? TTBound.Lower : TTBound.Exact);
            _table.Store(board.CurrentHash, depth, alpha, bestMove, bound);

            principalVariation.Add(bestMove);
            principalVariation.AddRange(bestChildPv);
            return alpha;
        }

        private int Quiescence(Board board, int depth, int alpha, int beta, CancellationToken cancellationToken)
        {
            if (ShouldStop(cancellationToken))
                return _evaluator.Evaluate(board);
            _nodes++;

            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
            var legalMoves = _generator.GenerateLegalMoves(
                board,
                skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);

            if (legalMoves.Count == 0)
                return -_options.MateScore;

            if (inCheck)
            {
                if (depth <= 0)
                    return _evaluator.Evaluate(board);

                foreach (var move in _moveOrdering.OrderMoves(board, legalMoves))
                {
                    board.Push(move.From, move.To);
                    try
                    {
                        int score = -Quiescence(board, Math.Max(0, depth - 1), -beta, -alpha, cancellationToken);
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

            if (TryFindImmediateMate(board, legalMoves, out _, cancellationToken))
                return _options.MateScore;

            if (OpponentHasImmediateMateAtLeaf(board, cancellationToken))
                return Math.Min(alpha, -_options.MateScore / 2);

            int standPat = _evaluator.Evaluate(board);
            if (standPat >= beta)
                return beta;
            if (standPat > alpha)
                alpha = standPat;
            if (depth <= 0)
                return alpha;

            var tacticalMoves = legalMoves
                .Where(move => board.GetPiece(move.To) != 0 || GivesCheck(board, move))
                .ToList();

            foreach (var move in _moveOrdering.OrderMoves(board, tacticalMoves))
            {
                board.Push(move.From, move.To);
                try
                {
                    int score = -Quiescence(board, depth - 1, -beta, -alpha, cancellationToken);
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

        private bool TryFindImmediateMate(Board board, List<Move> legalMoves, out Move mateMove, CancellationToken cancellationToken)
        {
            mateMove = default;
            foreach (var move in _moveOrdering.OrderMoves(board, legalMoves))
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
            opponentBoard.LoadState(board.GetState(), !board.IsRedTurn);
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

        private bool TryFindForcedCheckmate(Board board, int remainingPly, out Move mateMove, CancellationToken cancellationToken)
        {
            mateMove = default;
            if (remainingPly <= 0 || ShouldStop(cancellationToken))
                return false;

            var legalMoves = _generator.GenerateLegalMoves(
                board,
                skipPerpetualCheck: _options.SkipPerpetualCheckInsideSearch);

            foreach (var move in _moveOrdering.OrderMoves(board, legalMoves))
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
                    foreach (var reply in _moveOrdering.OrderMoves(board, replies))
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
                4 => 400,
                5 => 900,
                6 => 450,
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
