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
            get; private set;
        }
        public int N { get; private set; } = 0;

        public MCTSNode? Parent
        {
            get;
        }

        // 核心修复：多线程必须使用 ConcurrentDictionary
        public ConcurrentDictionary<Move, MCTSNode> Children { get; } = new ConcurrentDictionary<Move, MCTSNode>();
        public Move LastMove
        {
            get;
        }

        // 用于保护 N, W, Q 的线程锁
        private readonly object _lockObj = new object();

        public MCTSNode(MCTSNode? parent, double priorP, Move lastMove = default)
        {
            Parent = parent;
            P = priorP;
            LastMove = lastMove;
        }

        public bool IsLeaf => Children.IsEmpty;

        public double GetPUCTValue(double cPuct, int parentN)
        {
            double u = cPuct * P * Math.Sqrt(parentN) / (1 + N);
            return Q + u;
        }

        public void Expand(IEnumerable<(Move move, double prob)> policy)
        {
            foreach (var (move, prob) in policy)
            {
                // 并发安全的添加方式
                Children.TryAdd(move, new MCTSNode(this, prob, move));
            }
        }

        // 在类成员中定义
        private SpinLock _spinLock = new SpinLock();

        public void Update(double value)
        {
            bool lockTaken = false;
            try
            {
                // 尝试获取锁
                _spinLock.Enter(ref lockTaken);

                // 极速执行更新
                N++;
                W += value;
                Q = W / N;
            }
            finally
            {
                // 释放锁
                if (lockTaken)
                    _spinLock.Exit();
            }

            // 递归更新父节点（注意：Parent 的更新在锁外面，减少持有锁的时间）
            Parent?.Update(-value);
        }
    }
}