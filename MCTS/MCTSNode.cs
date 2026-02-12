using System;
using System.Collections.Generic;
using System.Linq;
using ChineseChessAI.Core;

namespace ChineseChessAI.MCTS
{
    public class MCTSNode
    {
        // --- AlphaZero 核心统计量 ---
        public double Q { get; private set; } = 0; // 节点的平均价值 (Mean Action Value)
        public double W { get; private set; } = 0; // 累计价值 (Total Action Value)
        public double P
        {
            get; private set;
        }      // 神经网络给出的先验概率 (Prior Probability)
        public int N { get; private set; } = 0;    // 访问次数 (Visit Count)

        // --- 树结构 ---
        public MCTSNode? Parent
        {
            get;
        }
        public Dictionary<Move, MCTSNode> Children { get; } = new Dictionary<Move, MCTSNode>();
        public Move LastMove
        {
            get;
        } // 导致到达此节点的那个动作

        public MCTSNode(MCTSNode? parent, double priorP, Move lastMove = default)
        {
            Parent = parent;
            P = priorP;
            LastMove = lastMove;
        }

        /// <summary>
        /// 判断是否为叶子节点（尚未展开）
        /// </summary>
        public bool IsLeaf => Children.Count == 0;

        /// <summary>
        /// 计算 PUCT 值：用于在搜索中选择最优分支
        /// 公式: U(s,a) = Q(s,a) + C_puct * P(s,a) * (sqrt(N_parent) / (1 + N_child))
        /// </summary>
        public double GetPUCTValue(double cPuct, int parentN)
        {
            // 如果节点从未被访问，Q 值为 0，但其 U 值会因为 P 和 parentN 而变得很大
            double u = cPuct * P * Math.Sqrt(parentN) / (1 + N);
            return Q + u;
        }

        /// <summary>
        /// 展开节点：根据神经网络预测的概率分布创建子节点
        /// </summary>
        /// <param name="policy">神经网络输出的策略向量 (已映射到合法走法)</param>
        public void Expand(IEnumerable<(Move move, double prob)> policy)
        {
            foreach (var (move, prob) in policy)
            {
                if (!Children.ContainsKey(move))
                {
                    Children[move] = new MCTSNode(this, prob, move);
                }
            }
        }

        /// <summary>
        /// 反向传播：更新路径上所有节点的统计数据
        /// </summary>
        /// <param name="value">神经网络预测的该局面的胜率 (-1 到 1)</param>
        public void Update(double value)
        {
            N++;
            W += value;
            Q = W / N;

            // 递归更新父节点（注意：在象棋中，由于是博弈，父节点的视角价值通常取反）
            Parent?.Update(-value);
        }
    }
}