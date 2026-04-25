using ChineseChessAI.Core;
using System.Collections.Concurrent;

namespace ChineseChessAI.MCTS
{
    public class MCTSNode
    {
        public double Q { get; private set; } = 0;
        public double W { get; private set; } = 0;
        public double P
        {
            get; set;
        }
        public int N { get; private set; } = 0;

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
            get;
        }
        public ConcurrentDictionary<Move, MCTSNode> Children { get; } = new ConcurrentDictionary<Move, MCTSNode>();
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

        public bool IsLeaf => Children.IsEmpty;

        public double GetPUCTValue(double cPuct, int parentN)
        {
            // 【核心修复】：快照式读取，确保在计算过程中 vl, n, w 是一致的
            int vl = VirtualLoss;
            int n_raw = N;
            double w_raw = W;

            int n = n_raw + vl;
            // Negamax 结构下，父节点评估子节点时需要取 -Q。
            // Virtual loss 必须让当前边临时变差，因此应加到子节点自身视角的 W 上。
            double q = n == 0 ? 0 : -(w_raw + vl) / n;
            double u = cPuct * P * Math.Sqrt(parentN) / (1 + n);
            return q + u;
        }

        public void Expand(IEnumerable<(Move move, double prob)> policy)
        {
            foreach (var (move, prob) in policy)
            {
                Children.TryAdd(move, new MCTSNode(this, prob, move));
            }
        }

        private SpinLock _spinLock = new SpinLock();

        public void Update(double value)
        {
            bool lockTaken = false;
            try
            {
                _spinLock.Enter(ref lockTaken);
                N++;
                W += value;
                Q = W / N;
            }
            finally
            {
                if (lockTaken)
                    _spinLock.Exit();
            }

            Parent?.Update(-value);
        }
    }
}
