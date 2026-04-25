using ChineseChessAI.NeuralNetwork;
using System.Collections.Concurrent;

namespace ChineseChessAI.MCTS
{
    internal static class InferenceService
    {
        private static readonly ConcurrentDictionary<CChessNet, SharedBatchWorker> Workers =
            new(ReferenceEqualityComparer.Instance);

        public static Lease Acquire(CChessNet model, int batchSize)
        {
            while (true)
            {
                var worker = Workers.GetOrAdd(model, static (m, size) => new SharedBatchWorker(m, size), batchSize);
                if (worker.TryAddReference())
                {
                    worker.EnsureBatchSize(batchSize);
                    return new Lease(model, worker);
                }

                Workers.TryRemove(new KeyValuePair<CChessNet, SharedBatchWorker>(model, worker));
            }
        }

        private static void Release(CChessNet model, SharedBatchWorker worker)
        {
            if (!worker.ReleaseReference())
            {
                return;
            }

            Workers.TryRemove(new KeyValuePair<CChessNet, SharedBatchWorker>(model, worker));
            worker.Dispose();
        }

        internal sealed class Lease : IDisposable
        {
            private readonly CChessNet _model;
            private SharedBatchWorker? _worker;

            internal Lease(CChessNet model, SharedBatchWorker worker)
            {
                _model = model;
                _worker = worker;
            }

            public Task<(float[] Policy, float Value)> PredictAsync(float[] inputData, CancellationToken cancellationToken = default)
            {
                var worker = _worker ?? throw new ObjectDisposedException(nameof(Lease));
                return worker.PredictAsync(inputData, cancellationToken);
            }

            public void Dispose()
            {
                var worker = Interlocked.Exchange(ref _worker, null);
                if (worker != null)
                {
                    Release(_model, worker);
                }
            }
        }

        internal sealed class SharedBatchWorker : IDisposable
        {
            private readonly BatchInference _batchInference;
            private int _referenceCount;
            private int _maxRequestedBatchSize;

            public SharedBatchWorker(CChessNet model, int batchSize)
            {
                _batchInference = new BatchInference(model, batchSize);
                _maxRequestedBatchSize = batchSize;
            }

            public bool TryAddReference()
            {
                while (true)
                {
                    int current = Volatile.Read(ref _referenceCount);
                    if (current < 0)
                    {
                        return false;
                    }

                    if (Interlocked.CompareExchange(ref _referenceCount, current + 1, current) == current)
                    {
                        return true;
                    }
                }
            }

            public bool ReleaseReference()
            {
                int remaining = Interlocked.Decrement(ref _referenceCount);
                if (remaining > 0)
                {
                    return false;
                }

                if (remaining == 0)
                {
                    Interlocked.Exchange(ref _referenceCount, -1);
                    return true;
                }

                throw new InvalidOperationException("InferenceService reference count dropped below zero.");
            }

            public void EnsureBatchSize(int batchSize)
            {
                if (batchSize > Volatile.Read(ref _maxRequestedBatchSize))
                {
                    Interlocked.Exchange(ref _maxRequestedBatchSize, batchSize);
                }
            }

            public Task<(float[] Policy, float Value)> PredictAsync(float[] inputData, CancellationToken cancellationToken = default)
            {
                return _batchInference.PredictAsync(inputData, cancellationToken);
            }

            public void Dispose()
            {
                _batchInference.Dispose();
            }
        }
    }
}
