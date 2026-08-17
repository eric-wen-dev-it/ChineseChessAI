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

    // 无锁线程安全置换表(Hyatt lockless XOR 法),支持 Lazy SMP 多线程并发读写:
    // 每个槽位存两个 ulong —— data(打包的条目)与 key(= hash ^ data)。读取时校验
    // key ^ data == hash,任何跨两半的撕裂读(一半来自旧写、一半来自新写)都会校验
    // 失败退化为未命中,绝不返回被撕裂拼接出的错误条目。x64 上对齐 ulong 读写原子,
    // 无需锁。空槽 data==0(真实条目分数经 +ScoreOffset 后 data 恒非零,故 0 唯一表示空)。
    public sealed class TranspositionTable
    {
        private const int MateScoreWindow = 10_000;

        // 条目打包到单个 ulong:
        //   bits  0-6  : from (0-89)
        //   bits  7-13 : to   (0-89)
        //   bits 14-15 : bound (Exact/Lower/Upper)
        //   bits 16-24 : depth (0-511,已 clamp)
        //   bits 25-46 : score + ScoreOffset (22 位,覆盖 ±2,000,000)
        private const int ScoreOffset = 1 << 21; // 2,097,152
        private const int ScoreClamp = 2_000_000;

        private readonly ulong[] _keys;
        private readonly ulong[] _data;

        public TranspositionTable(int entryCount)
        {
            int size = Math.Max(1024, entryCount);
            _keys = new ulong[size];
            _data = new ulong[size];
        }

        public bool TryGet(ulong hash, int ply, int mateScore, out TTEntry entry)
        {
            int index = GetIndex(hash);
            ulong data = _data[index];
            if (data == 0)
            {
                entry = default;
                return false;
            }

            ulong key = _keys[index];
            if ((key ^ data) != hash)
            {
                // 撕裂读或哈希冲突:视为未命中。
                entry = default;
                return false;
            }

            Unpack(data, out int depth, out int rawScore, out Move move, out TTBound bound);
            entry = new TTEntry
            {
                Hash = hash,
                Depth = depth,
                Score = ScoreFromTable(rawScore, ply, mateScore),
                BestMove = move,
                Bound = bound
            };
            return true;
        }

        public void Store(ulong hash, int depth, int score, Move bestMove, TTBound bound, int ply, int mateScore)
        {
            int index = GetIndex(hash);
            ulong existing = _data[index];
            if (existing != 0)
            {
                ulong existingKey = _keys[index];
                ulong existingHash = existingKey ^ existing;
                // 不同局面占用同一槽且既有条目更深:保留深条目(浅条目信息量小)。
                if (existingHash != hash && ((existing >> 16) & 0x1FF) > (uint)Math.Clamp(depth, 0, 511))
                    return;
            }

            ulong packed = Pack(depth, ScoreToTable(score, ply, mateScore), bestMove, bound);
            // 先写 data 再写 key:读方先读 data(非 0 才继续),再读 key 校验;
            // 顺序不影响正确性(XOR 校验兜底),此序减少读方看到"新 key 配旧 data"。
            _data[index] = packed;
            _keys[index] = hash ^ packed;
        }

        // 静搜条目固定深度 0,且绝不覆盖任何深度≥1 的既有条目(主搜索条目信息量更大)。
        public void StoreQuiescence(ulong hash, int score, Move bestMove, TTBound bound, int ply, int mateScore)
        {
            int index = GetIndex(hash);
            ulong existing = _data[index];
            if (existing != 0 && ((existing >> 16) & 0x1FF) > 0)
                return;

            ulong packed = Pack(0, ScoreToTable(score, ply, mateScore), bestMove, bound);
            _data[index] = packed;
            _keys[index] = hash ^ packed;
        }

        private int GetIndex(ulong hash) => (int)(hash % (ulong)_data.Length);

        private static ulong Pack(int depth, int rawScore, Move move, TTBound bound)
        {
            ulong from = (ulong)(move.From & 0x7F);
            ulong to = (ulong)(move.To & 0x7F);
            ulong b = (ulong)((int)bound & 0x3);
            ulong d = (ulong)(Math.Clamp(depth, 0, 511) & 0x1FF);
            ulong s = (ulong)(Math.Clamp(rawScore, -ScoreClamp, ScoreClamp) + ScoreOffset) & 0x3FFFFF;
            return from | (to << 7) | (b << 14) | (d << 16) | (s << 25);
        }

        private static void Unpack(ulong data, out int depth, out int rawScore, out Move move, out TTBound bound)
        {
            int from = (int)(data & 0x7F);
            int to = (int)((data >> 7) & 0x7F);
            bound = (TTBound)((int)((data >> 14) & 0x3));
            depth = (int)((data >> 16) & 0x1FF);
            rawScore = (int)((data >> 25) & 0x3FFFFF) - ScoreOffset;
            move = new Move(from, to);
        }

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
