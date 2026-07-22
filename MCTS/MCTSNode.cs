using ChineseChessAI.Core;
namespace ChineseChessAI.MCTS
{
    public class MCTSNode
    {
        private double _q = 0;
        private double _w = 0;
        private int _n = 0;

        public double Q => Volatile.Read(ref _q);
        public double W => Volatile.Read(ref _w);
        public double P
        {
            get; set;
        }
        public int N => Volatile.Read(ref _n);

        // 【核心修复】：增加虚拟损失字段，用于多线程防碰撞
        public int VirtualLoss = 0;

        // 【核心修复】：防止重复展开的原子标志
        private int _isExpanding = 0;

        public bool TryMarkExpanding()
        {
            return Interlocked.CompareExchange(ref _isExpanding, 1, 0) == 0;
        }

        public void UnmarkExpanding()
        {
            _isExpanding = 0;
        }

        public MCTSNode? Parent
        {
            get; private set;
        }
        private MCTSChild[] _children = Array.Empty<MCTSChild>();
        public IReadOnlyList<MCTSChild> Children => Volatile.Read(ref _children);
        public Move LastMove
        {
            get;
        }

        public MCTSNode(MCTSNode? parent, double priorP, Move lastMove = default)
        {
            Parent = parent;
            P = priorP;
            LastMove = lastMove;
        }

        public bool IsLeaf => Volatile.Read(ref _children).Length == 0;

        public bool IsExpanding => Volatile.Read(ref _isExpanding) != 0;

        public double GetPUCTValue(double cPuct, int parentN)
        {
            // Volatile reads are intentionally lightweight; PUCT tolerates minor concurrent drift.
            int vl = Volatile.Read(ref VirtualLoss);
            int n_raw = Volatile.Read(ref _n);
            double w_raw = Volatile.Read(ref _w);

            int n = n_raw + vl;
            // Negamax 结构下，父节点评估子节点时需要取 -Q。
            // 未访问节点用父节点均值的轻微保守估计，避免 FPU=0 过早压低其它先验较好的候选。
            double q = n == 0 ? (Parent?.Q ?? 0.0) - 0.20 : -(w_raw + vl) / n;
            double u = cPuct * P * Math.Sqrt(parentN) / (1 + n);
            return q + u;
        }

        public void Expand(IEnumerable<(Move move, double prob)> policy)
        {
            var children = policy
                .GroupBy(x => x.move)
                .Select(group =>
                {
                    var (move, prob) = group.First();
                    return new MCTSChild(move, new MCTSNode(this, prob, move));
                })
                .ToArray();

            Volatile.Write(ref _children, children);
        }

        public bool TryGetChild(Move move, out MCTSNode child)
        {
            foreach (var candidate in Volatile.Read(ref _children))
            {
                if (candidate.Move.Equals(move))
                {
                    child = candidate.Node;
                    return true;
                }
            }

            child = default!;
            return false;
        }

        public void DetachParent()
        {
            Parent = null;
        }

        private SpinLock _spinLock = new SpinLock();

        public void Update(double value)
        {
            if (!double.IsFinite(value))
                value = 0.0;

            bool lockTaken = false;
            try
            {
                _spinLock.Enter(ref lockTaken);
                _n++;
                _w += value;
                _q = _w / _n;
            }
            finally
            {
                if (lockTaken)
                    _spinLock.Exit();
            }

            Parent?.Update(-value);
        }
    }

    public readonly record struct MCTSChild(Move Move, MCTSNode Node);
}
