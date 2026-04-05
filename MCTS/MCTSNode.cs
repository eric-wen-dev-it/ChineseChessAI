using ChineseChessAI.Core;
using System.Collections.Concurrent;
using System.Threading;

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
            // 【核心修复】：计算 UCB 时，将虚拟损失视作被访问过且输掉了一局
            int n = N + VirtualLoss;
            double q = n == 0 ? 0 : (W - VirtualLoss) / n;
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