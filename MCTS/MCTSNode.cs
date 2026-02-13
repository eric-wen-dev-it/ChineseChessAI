using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using ChineseChessAI.Core;

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

        public void Update(double value)
        {
            // 核心修复：防止并发更新导致的脏数据
            lock (_lockObj)
            {
                N++;
                W += value;
                Q = W / N;
            }
            Parent?.Update(-value);
        }
    }
}