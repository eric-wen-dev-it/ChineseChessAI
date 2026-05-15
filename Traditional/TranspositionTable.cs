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
        private const int MateScoreWindow = 10_000;
        private readonly TTEntry[] _entries;

        public TranspositionTable(int entryCount)
        {
            _entries = new TTEntry[Math.Max(1024, entryCount)];
        }

        public bool TryGet(ulong hash, int ply, int mateScore, out TTEntry entry)
        {
            entry = _entries[GetIndex(hash)];
            if (entry.Hash != hash)
                return false;

            entry = entry with
            {
                Score = ScoreFromTable(entry.Score, ply, mateScore)
            };
            return true;
        }

        public void Store(ulong hash, int depth, int score, Move bestMove, TTBound bound, int ply, int mateScore)
        {
            int index = GetIndex(hash);
            var existing = _entries[index];
            if (existing.Hash != 0 && existing.Hash != hash && existing.Depth > depth)
                return;

            _entries[index] = new TTEntry
            {
                Hash = hash,
                Depth = depth,
                Score = ScoreToTable(score, ply, mateScore),
                BestMove = bestMove,
                Bound = bound
            };
        }

        private int GetIndex(ulong hash) => (int)(hash % (ulong)_entries.Length);

        private static int ScoreToTable(int score, int ply, int mateScore)
        {
            if (score >= mateScore - MateScoreWindow)
                return score + ply;
            if (score <= -mateScore + MateScoreWindow)
                return score - ply;
            return score;
        }

        private static int ScoreFromTable(int score, int ply, int mateScore)
        {
            if (score >= mateScore - MateScoreWindow)
                return score - ply;
            if (score <= -mateScore + MateScoreWindow)
                return score + ply;
            return score;
        }
    }
}
