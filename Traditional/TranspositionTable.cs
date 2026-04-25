using ChineseChessAI.Core;

namespace ChineseChessAI.Traditional
{
    public readonly struct TTEntry
    {
        public ulong Hash { get; init; }
        public int Depth { get; init; }
        public int Score { get; init; }
        public Move BestMove { get; init; }
        public TTBound Bound { get; init; }
    }

    public sealed class TranspositionTable
    {
        private readonly TTEntry[] _entries;

        public TranspositionTable(int entryCount)
        {
            _entries = new TTEntry[Math.Max(1024, entryCount)];
        }

        public bool TryGet(ulong hash, out TTEntry entry)
        {
            entry = _entries[GetIndex(hash)];
            return entry.Hash == hash;
        }

        public void Store(ulong hash, int depth, int score, Move bestMove, TTBound bound)
        {
            int index = GetIndex(hash);
            var existing = _entries[index];
            if (existing.Hash != 0 && existing.Hash != hash && existing.Depth > depth)
                return;

            _entries[index] = new TTEntry
            {
                Hash = hash,
                Depth = depth,
                Score = score,
                BestMove = bestMove,
                Bound = bound
            };
        }

        private int GetIndex(ulong hash) => (int)(hash % (ulong)_entries.Length);
    }
}
