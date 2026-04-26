using ChineseChessAI.Core;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System.Collections.Concurrent;
using static TorchSharp.torch;

namespace ChineseChessAI.MCTS
{
    public class MCTSEngine : IDisposable
    {
        private const int LegalMovesCacheCapacity = 32768;
        private const int InferenceCacheCapacity = 16384;

        private readonly CChessNet _model;
        private readonly ChineseChessRuleEngine _rules;
        private readonly InferenceService.Lease _inferenceLease;
        private readonly double _cPuct;
        private readonly BoundedCache<Move[]> _legalMovesCache = new(LegalMovesCacheCapacity);
        private readonly BoundedCache<CachedInference> _inferenceCache = new(InferenceCacheCapacity);
        private readonly ConcurrentDictionary<ulong, Lazy<Task<CachedInference>>> _inferenceInFlight = new();

        private static readonly RuntimeDiagnostics.RollingCounter LegalMovesCacheHitCounter = new("LegalMovesCacheHit", 500);
        private static readonly RuntimeDiagnostics.RollingCounter LegalMovesCacheMissCounter = new("LegalMovesCacheMiss", 500);
        private static readonly RuntimeDiagnostics.RollingCounter InferenceCacheHitCounter = new("InferenceCacheHit", 200);
        private static readonly RuntimeDiagnostics.RollingCounter InferenceCacheMissCounter = new("InferenceCacheMiss", 200);
        private static readonly RuntimeDiagnostics.RollingCounter CloneBoardCounter = new("CloneBoardCalls", 2000);
        private static readonly RuntimeDiagnostics.RollingCounter PushCounter = new("BoardPushCalls", 5000);
        private static readonly RuntimeDiagnostics.RollingCounter PopCounter = new("BoardPopCalls", 5000);
        private static readonly RuntimeDiagnostics.RollingCounter LegalMovesCacheTrimCounter = new("LegalMovesCacheTrim", 50);
        private static readonly RuntimeDiagnostics.RollingCounter InferenceCacheTrimCounter = new("InferenceCacheTrim", 50);

        public MCTSEngine(CChessNet model, int batchSize = 64, double cPuct = 2.5, Action<string>? statusChanged = null)
        {
            _model = model;
            _cPuct = cPuct;
            _rules = new ChineseChessRuleEngine(new MoveGenerator(statusChanged));

            _model.eval();
            _inferenceLease = InferenceService.Acquire(_model, batchSize);
        }

        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations, int currentMoves = 0, int maxMoves = 999, CancellationToken cancellationToken = default, bool addRootNoise = true)
        {
            var root = new MCTSNode(null, 1.0);
            await SearchAsync(root, CloneBoard(board), currentMoves, maxMoves, 0, cancellationToken);

            if (addRootNoise)
            {
                ApplyDirichletNoise(root);
            }

            int numThreads = 8;
            int baseSims = (simulations - 1) / numThreads;
            int extraSims = (simulations - 1) % numThreads;

            var tasks = Enumerable.Range(0, numThreads).Select(t => Task.Run(async () =>
            {
                int taskSims = (t == 0) ? baseSims + extraSims : baseSims;
                for (int i = 0; i < taskSims; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    await SearchAsync(root, CloneBoard(board), currentMoves, maxMoves, 0, cancellationToken);
                }
            }));

            await Task.WhenAll(tasks);

            float[] piData = new float[8100];
            var legalMoves = GetLegalMoves(board, board.CurrentHash, cancellationToken);

            if (root.Children.IsEmpty)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (legalMoves.Count == 0)
                {
                    throw new Exception("无合法走法");
                }

                RuntimeDiagnostics.Log($"[MCTS空展开兜底] legalMoves={legalMoves.Count}, simulations={simulations}, currentMoves={currentMoves}, maxMoves={maxMoves}, canceled={cancellationToken.IsCancellationRequested}");
                return CreateFallbackMovePolicy(legalMoves);
            }

            var rootChildren = root.Children.ToArray();
            double totalVisits = rootChildren.Sum(x => x.Value.N);

            foreach (var child in rootChildren)
            {
                int moveIdx = child.Key.ToNetworkIndex();
                if (moveIdx >= 0 && moveIdx < 8100)
                {
                    piData[moveIdx] = (float)(child.Value.N / (totalVisits > 0 ? totalVisits : 1));
                }
            }

            var bestMove = rootChildren.OrderByDescending(x => x.Value.N).First().Key;
            return (bestMove, piData);
        }

        private async Task SearchAsync(MCTSNode node, Board board, int currentMoves, int maxMoves, int depth, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (currentMoves + depth >= maxMoves)
            {
                float advantage = BoardEvaluation.GetBoardAdvantage(board);
                node.Update(board.IsRedTurn ? advantage * 0.5 : -advantage * 0.5);
                return;
            }

            if (node.IsLeaf)
            {
                if (!node.TryMarkExpanding())
                {
                    return;
                }

                try
                {
                    ulong boardHash = board.CurrentHash;
                    var legalMoves = GetLegalMoves(board, boardHash, cancellationToken);
                    if (legalMoves.Count == 0)
                    {
                        node.Update(-1.0);
                        return;
                    }

                    var inference = await GetInferenceAsync(board, boardHash, cancellationToken);
                    var rawPolicy = GetFilteredPolicy(inference.PolicyLogits, legalMoves).ToList();
                    node.Expand(rawPolicy);
                    node.Update(inference.Value);
                }
                finally
                {
                    node.UnmarkExpanding();
                }

                return;
            }

            var childrenSnapshot = node.Children.ToArray();
            var bestChild = childrenSnapshot
                .OrderByDescending(x => x.Value.GetPUCTValue(_cPuct, node.N))
                .First();

            Interlocked.Increment(ref bestChild.Value.VirtualLoss);
            PushCounter.AddSample(1);
            board.Push(bestChild.Key.From, bestChild.Key.To);

            try
            {
                await SearchAsync(bestChild.Value, board, currentMoves, maxMoves, depth + 1, cancellationToken);
            }
            finally
            {
                PopCounter.AddSample(1);
                board.Pop();
                Interlocked.Decrement(ref bestChild.Value.VirtualLoss);
            }
        }

        private void ApplyDirichletNoise(MCTSNode root)
        {
            if (root.Children.IsEmpty)
            {
                return;
            }

            const double epsilon = 0.25;
            const double alpha = 0.3;
            var childrenSnapshot = root.Children.ToArray();
            var noise = SampleDirichlet(childrenSnapshot.Length, alpha);
            for (int i = 0; i < childrenSnapshot.Length; i++)
            {
                var node = childrenSnapshot[i].Value;
                node.P = (1 - epsilon) * node.P + epsilon * noise[i];
            }
        }

        private double[] SampleDirichlet(int count, double alpha)
        {
            double[] samples = new double[count];
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                double sample = GammaSample(alpha, 1.0);
                samples[i] = sample;
                sum += sample;
            }

            for (int i = 0; i < count; i++)
            {
                samples[i] /= sum;
            }

            return samples;
        }

        private double GammaSample(double alpha, double beta)
        {
            if (alpha < 1.0)
            {
                return GammaSample(alpha + 1.0, beta) * Math.Pow(Random.Shared.NextDouble(), 1.0 / alpha);
            }

            double d = alpha - 1.0 / 3.0;
            double c = 1.0 / Math.Sqrt(9.0 * d);
            while (true)
            {
                double x;
                double v;
                double u = Random.Shared.NextDouble();
                do
                {
                    x = NormalSample();
                    v = 1.0 + c * x;
                } while (v <= 0);

                v = v * v * v;
                if (u < 1.0 - 0.0331 * x * x * x * x)
                {
                    return d * v / beta;
                }

                if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
                {
                    return d * v / beta;
                }
            }
        }

        private double NormalSample()
        {
            double u1 = Math.Max(1e-10, 1.0 - Random.Shared.NextDouble());
            double u2 = Random.Shared.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        private IEnumerable<(Move move, double prob)> GetFilteredPolicy(float[] logits, List<Move> legalMoves)
        {
            var probs = new List<(Move, double)>();
            float maxLogit = float.MinValue;
            var validMoves = new List<(Move, float)>();

            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                {
                    float val = logits[idx];
                    if (!float.IsFinite(val))
                    {
                        continue;
                    }

                    if (val > maxLogit)
                    {
                        maxLogit = val;
                    }

                    validMoves.Add((move, val));
                }
            }

            if (validMoves.Count == 0)
            {
                double uniform = legalMoves.Count > 0 ? 1.0 / legalMoves.Count : 0.0;
                return legalMoves.Select(move => (move, uniform));
            }

            double sum = 0;
            foreach (var (move, val) in validMoves)
            {
                double p = Math.Exp(val - maxLogit);
                if (!double.IsFinite(p))
                {
                    continue;
                }

                probs.Add((move, p));
                sum += p;
            }

            if (probs.Count == 0 || !double.IsFinite(sum) || sum <= 0)
            {
                double uniform = legalMoves.Count > 0 ? 1.0 / legalMoves.Count : 0.0;
                return legalMoves.Select(move => (move, uniform));
            }

            return probs.Select(x => (x.Item1, x.Item2 / sum));
        }

        private (Move move, float[] pi) CreateFallbackMovePolicy(List<Move> legalMoves)
        {
            float[] piData = new float[8100];
            float probability = legalMoves.Count > 0 ? 1.0f / legalMoves.Count : 0f;
            foreach (var move in legalMoves)
            {
                int moveIdx = move.ToNetworkIndex();
                if (moveIdx >= 0 && moveIdx < piData.Length)
                {
                    piData[moveIdx] = probability;
                }
            }

            Move selectedMove = legalMoves[Random.Shared.Next(legalMoves.Count)];
            return (selectedMove, piData);
        }

        private Board CloneBoard(Board original)
        {
            CloneBoardCounter.AddSample(1);
            return original.Clone();
        }

        private List<Move> GetLegalMoves(Board board, ulong boardHash, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_legalMovesCache.TryGet(boardHash, out Move[] cachedMoves))
            {
                LegalMovesCacheHitCounter.AddSample(1);
                return new List<Move>(cachedMoves);
            }

            LegalMovesCacheMissCounter.AddSample(1);
            var legalMoves = _rules.GetLegalMoves(board, cancellationToken: cancellationToken);
            if (_legalMovesCache.TryAdd(boardHash, legalMoves.ToArray()))
            {
                TrimLegalMovesCacheIfNeeded();
            }

            return legalMoves;
        }

        private async Task<CachedInference> GetInferenceAsync(Board board, ulong boardHash, CancellationToken cancellationToken)
        {
            if (_inferenceCache.TryGet(boardHash, out CachedInference cached))
            {
                InferenceCacheHitCounter.AddSample(1);
                return cached;
            }

            InferenceCacheMissCounter.AddSample(1);
            Lazy<Task<CachedInference>> lazyTask = _inferenceInFlight.GetOrAdd(
                boardHash,
                _ => CreateInferenceTask(board, boardHash));

            CachedInference result = await lazyTask.Value.WaitAsync(cancellationToken).ConfigureAwait(false);
            if (_inferenceCache.TryAdd(boardHash, result))
            {
                TrimInferenceCacheIfNeeded();
            }

            return result;
        }

        private Lazy<Task<CachedInference>> CreateInferenceTask(Board board, ulong boardHash)
        {
            return new Lazy<Task<CachedInference>>(() =>
            {
                Task<CachedInference> task = EvaluateBoardAsync(board, CancellationToken.None);
                task.ContinueWith(
                    _ => _inferenceInFlight.TryRemove(boardHash, out Lazy<Task<CachedInference>> _),
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);

                return task;
            }, LazyThreadSafetyMode.ExecutionAndPublication);
        }

        private async Task<CachedInference> EvaluateBoardAsync(Board board, CancellationToken cancellationToken)
        {
            float[] inputData;
            bool isRed = board.IsRedTurn;

            using (var tensor = StateEncoder.Encode(board))
            using (var inputFloat = tensor.to_type(ScalarType.Float32))
            {
                inputData = inputFloat.data<float>().ToArray();
            }

            var (policyLogits, value) = await _inferenceLease.PredictAsync(inputData, cancellationToken).ConfigureAwait(false);
            if (!isRed)
            {
                policyLogits = StateEncoder.FlipPolicy(policyLogits);
            }

            return new CachedInference(policyLogits, value);
        }

        private void TrimLegalMovesCacheIfNeeded()
        {
            int removed = _legalMovesCache.TrimIfNeeded();
            if (removed > 0)
            {
                LegalMovesCacheTrimCounter.AddSample(removed);
            }
        }

        private void TrimInferenceCacheIfNeeded()
        {
            int removed = _inferenceCache.TrimIfNeeded();
            if (removed > 0)
            {
                InferenceCacheTrimCounter.AddSample(removed);
            }
        }

        public void Dispose()
        {
            _inferenceLease?.Dispose();
        }

        private sealed record CachedInference(float[] PolicyLogits, double Value);

        private sealed class BoundedCache<TValue> where TValue : class
        {
            private readonly int _capacity;
            private readonly ConcurrentDictionary<ulong, CacheEntry<TValue>> _entries = new();
            private long _tick;
            private int _trimGate;

            public BoundedCache(int capacity)
            {
                _capacity = capacity;
            }

            public bool TryGet(ulong key, out TValue value)
            {
                if (_entries.TryGetValue(key, out CacheEntry<TValue>? entry))
                {
                    entry.Touch(Interlocked.Increment(ref _tick));
                    value = entry.Value;
                    return true;
                }

                value = default!;
                return false;
            }

            public bool TryAdd(ulong key, TValue value)
            {
                long tick = Interlocked.Increment(ref _tick);
                return _entries.TryAdd(key, new CacheEntry<TValue>(value, tick));
            }

            public int TrimIfNeeded()
            {
                if (_entries.Count <= _capacity)
                {
                    return 0;
                }

                if (Interlocked.CompareExchange(ref _trimGate, 1, 0) != 0)
                {
                    return 0;
                }

                try
                {
                    if (_entries.Count <= _capacity)
                    {
                        return 0;
                    }

                    KeyValuePair<ulong, CacheEntry<TValue>>[] entrySnapshot = _entries.ToArray();
                    int targetCount = _capacity - Math.Max(1, _capacity / 8);
                    int toRemove = Math.Max(1, entrySnapshot.Length - targetCount);
                    var victims = entrySnapshot
                        .OrderBy(kvp => kvp.Value.LastAccessTick)
                        .Take(toRemove)
                        .Select(kvp => kvp.Key)
                        .ToArray();

                    int removed = 0;
                    foreach (ulong victim in victims)
                    {
                        if (_entries.TryRemove(victim, out _))
                        {
                            removed++;
                        }
                    }

                    return removed;
                }
                finally
                {
                    Volatile.Write(ref _trimGate, 0);
                }
            }
        }

        private sealed class CacheEntry<TValue> where TValue : class
        {
            public CacheEntry(TValue value, long lastAccessTick)
            {
                Value = value;
                LastAccessTick = lastAccessTick;
            }

            public TValue Value
            {
                get;
            }
            public long LastAccessTick;

            public void Touch(long tick)
            {
                Volatile.Write(ref LastAccessTick, tick);
            }
        }
    }
}
