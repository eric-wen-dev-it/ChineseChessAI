using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using ChineseChessAI.Core;
using ChineseChessAI.NeuralNetwork;

namespace ChineseChessAI.MCTS
{
    public class MCTSEngine
    {
        private readonly CChessNet _model;
        private readonly MoveGenerator _generator;
        private readonly double _cPuct = 1.5; // 控制探索与开发的平衡系数

        public MCTSEngine(CChessNet model)
        {
            _model = model;
            _generator = new MoveGenerator();
        }

        /// <summary>
        /// 执行 MCTS 搜索并返回最佳走法
        /// </summary>
        /// <param name="board">当前棋盘状态</param>
        /// <param name="simulations">模拟次数 (例如 800 或 1600)</param>
        public Move GetBestMove(Board board, int simulations)
        {
            // 1. 初始化根节点
            var root = new MCTSNode(null, 1.0);

            for (int i = 0; i < simulations; i++)
            {
                // 每次模拟都从原始棋盘的副本开始
                Board tempBoard = CloneBoard(board);
                Search(root, tempBoard);
            }

            // 2. 搜索完成后，选择访问次数 N 最高的动作作为最佳走法
            return root.Children.OrderByDescending(x => x.Value.N).First().Key;
        }

        private void Search(MCTSNode node, Board board)
        {
            // A. 选择 (Selection)
            // 如果当前节点已经展开，根据 PUCT 公式向下递归
            if (!node.IsLeaf)
            {
                var bestChild = node.Children
                    .OrderByDescending(x => x.Value.GetPUCTValue(_cPuct, node.N))
                    .First();

                board.Push(bestChild.Key.From, bestChild.Key.To);
                Search(bestChild.Value, board);
                return;
            }

            // B. 扩展与评估 (Expansion & Evaluation)
            // 1. 将局面转为 Tensor
            var inputTensor = StateEncoder.Encode(board);

            // 2. 神经网络推理
            _model.eval();
            using (var noGrad = torch.no_grad())
            {
                var (policyLogits, valueTensor) = _model.forward(inputTensor);

                // 获取当前局面的胜率评估 z
                double value = valueTensor.item<float>();

                // 获取所有合法动作
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count == 0) // 处理绝杀或困毙
                {
                    node.Update(-1.0); // 当前玩家输了
                    return;
                }

                // 3. 提取合法动作对应的 Policy 概率并进行 Softmax 归一化
                var filteredPolicy = GetFilteredPolicy(policyLogits, legalMoves);

                // 4. 展开节点
                node.Expand(filteredPolicy);

                // C. 反向传播 (Backpropagation)
                node.Update(value);
            }
        }

        /// <summary>
        /// 执行 MCTS 搜索并返回最佳走法以及对应的访问频率分布 (π)
        /// </summary>
        public (Move move, torch.Tensor pi) GetMoveWithProbabilities(Board board, int simulations)
        {
            // 1. 执行 MCTS 搜索逻辑
            var root = new MCTSNode(null, 1.0);
            for (int i = 0; i < simulations; i++)
            {
                // 实际开发中建议实现 Board 的 Clone 或从状态快照恢复
                // 这里简化演示
                Search(root, board);
            }

            // 2. 构造 2086 维的概率向量 π
            float[] piData = new float[2086];
            double totalVisits = root.Children.Sum(x => x.Value.N);

            foreach (var child in root.Children)
            {
                int moveIdx = child.Key.ToNetworkIndex();
                // 概率计算公式: pi = N_i / Total_N
                piData[moveIdx] = (float)(child.Value.N / totalVisits);
            }

            var piTensor = torch.tensor(piData, new long[] { 8100 });

            // 3. 选择访问次数最多的走法作为实际执行的动作
            var bestMove = root.Children.OrderByDescending(x => x.Value.N).First().Key;

            return (bestMove, piTensor);
        }
        private IEnumerable<(Move move, double prob)> GetFilteredPolicy(torch.Tensor logits, List<Move> legalMoves)
        {
            // 简单实现：只取合法动作对应的原始概率并归一化
            // 在生产环境中，建议在 GPU 上直接通过 Mask 处理
            var probs = new List<(Move, double)>();
            double sum = 0;

            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                // 这里的索引逻辑需与 Move.cs 保持一致
                float val = Math.Max(0, logits[0, idx].item<float>());
                double p = Math.Exp(val); // 简易 Softmax
                probs.Add((move, p));
                sum += p;
            }

            return probs.Select(x => (x.Item1, x.Item2 / sum));
        }

        
        private Board CloneBoard(Board original)
        {
            var newBoard = new Board();
            // 必须要将原始棋盘的棋子位置复制过去，否则 Search 会报错
            newBoard.LoadState(original.GetState(), original.IsRedTurn);
            return newBoard;
        }
    }
}