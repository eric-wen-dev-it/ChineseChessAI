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
        public Move GetBestMove(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            for (int i = 0; i < simulations; i++)
            {
                Board tempBoard = CloneBoard(board);
                Search(root, tempBoard);
            }

            // 防御性检查：确保有子节点
            if (root.Children.Count == 0)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count == 0)
                    throw new Exception("当前局面无合法走法。");
                return legalMoves[0];
            }

            return root.Children.OrderByDescending(x => x.Value.N).First().Key;
        }

        private void Search(MCTSNode node, Board board)
        {
            try
            {
                // A. 选择 (Selection)
                if (!node.IsLeaf)
                {
                    if (node.Children.Count == 0)
                        return;

                    var bestChild = node.Children
                        .OrderByDescending(x => x.Value.GetPUCTValue(_cPuct, node.N))
                        .First();

                    board.Push(bestChild.Key.From, bestChild.Key.To);
                    Search(bestChild.Value, board);
                    return;
                }

                // B. 扩展与评估 (Expansion & Evaluation)
                using (var scope = torch.NewDisposeScope())
                {
                    var inputTensor = StateEncoder.Encode(board);
                    _model.eval();

                    using (var noGrad = torch.no_grad())
                    {
                        var (policyLogits, valueTensor) = _model.forward(inputTensor);

                        double value = valueTensor.item<float>();

                        var legalMoves = _generator.GenerateLegalMoves(board);
                        if (legalMoves.Count == 0)
                        {
                            node.Update(-1.0); // 处理绝杀或困毙
                            return;
                        }

                        var filteredPolicy = GetFilteredPolicy(policyLogits, legalMoves);
                        node.Expand(filteredPolicy);

                        // C. 反向传播 (Backpropagation)
                        node.Update(value);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MCTS Search Error] {ex.Message}");
            }
        }

        /// <summary>
        /// 执行 MCTS 搜索并返回最佳走法以及对应的概率分布 (π)
        /// </summary>
        public (Move move, torch.Tensor pi) GetMoveWithProbabilities(Board board, int simulations)
        {
            torch.Tensor piResult;
            Move bestMove;

            // 使用局部作用域管理模拟过程中的临时 Tensor
            using (var scope = torch.NewDisposeScope())
            {
                var root = new MCTSNode(null, 1.0);
                for (int i = 0; i < simulations; i++)
                {
                    Board tempBoard = CloneBoard(board);
                    Search(root, tempBoard);
                }

                if (root.Children.Count == 0)
                {
                    var legalMoves = _generator.GenerateLegalMoves(board);
                    if (legalMoves.Count == 0)
                        throw new Exception("无合法走法");

                    // 关键：detach() 确保 Tensor 脱离 DisposeScope 的自动释放列表
                    return (legalMoves[0], torch.zeros(new long[] { 8100 }).detach());
                }

                float[] piData = new float[8100];
                double totalVisits = root.Children.Sum(x => x.Value.N);
                if (totalVisits == 0)
                    totalVisits = 1;

                foreach (var child in root.Children)
                {
                    int moveIdx = child.Key.ToNetworkIndex();
                    if (moveIdx >= 0 && moveIdx < 8100)
                        piData[moveIdx] = (float)(child.Value.N / totalVisits);
                }

                // 核心修复：创建 Tensor 后立即执行 .detach()
                // 此时句柄在 scope 结束后依然有效，供 SelfPlay 调用 .data<float>()
                piResult = torch.tensor(piData, new long[] { 8100 }).detach();

                bestMove = root.Children.OrderByDescending(x => x.Value.N).First().Key;
            }

            return (bestMove, piResult);
        }

        private IEnumerable<(Move move, double prob)> GetFilteredPolicy(torch.Tensor logits, List<Move> legalMoves)
        {
            var probs = new List<(Move, double)>();

            // 1. 寻找合法走法中的最大 Logit (用于数值稳定)
            float maxLogit = float.MinValue;
            var validMoves = new List<(Move, float)>();

            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                {
                    float val = logits[0, idx].item<float>();
                    if (val > maxLogit)
                        maxLogit = val;
                    validMoves.Add((move, val));
                }
            }

            // 2. 计算 Stable Softmax (每个值减去最大值再求 Exp，永不溢出)
            double sum = 0;
            foreach (var (move, val) in validMoves)
            {
                double p = Math.Exp(val - maxLogit);
                probs.Add((move, p));
                sum += p;
            }

            if (sum == 0)
                sum = 1;
            return probs.Select(x => (x.Item1, x.Item2 / sum));
        }

        private Board CloneBoard(Board original)
        {
            var newBoard = new Board();
            newBoard.LoadState(original.GetState(), original.IsRedTurn);
            return newBoard;
        }

        /// <summary>
        /// 新增方法：直接返回 C# 数组，彻底解决多线程 Tensor 生命周期问题
        /// </summary>
        public (Move move, float[] pi) GetMoveWithProbabilitiesAsArray(Board board, int simulations)
        {
            using (var scope = torch.NewDisposeScope())
            {
                var root = new MCTSNode(null, 1.0);
                for (int i = 0; i < simulations; i++)
                {
                    Board tempBoard = CloneBoard(board);
                    Search(root, tempBoard);
                }

                if (root.Children.Count == 0)
                {
                    var legalMoves = _generator.GenerateLegalMoves(board);
                    return (legalMoves[0], new float[8100]);
                }

                float[] piData = new float[8100];
                double totalVisits = root.Children.Sum(x => x.Value.N);
                if (totalVisits == 0)
                    totalVisits = 1;

                foreach (var child in root.Children)
                {
                    int moveIdx = child.Key.ToNetworkIndex();
                    if (moveIdx >= 0 && moveIdx < 8100)
                        piData[moveIdx] = (float)(child.Value.N / totalVisits);
                }

                var bestMove = root.Children.OrderByDescending(x => x.Value.N).First().Key;

                // 返回纯数据数组，不返回 Tensor 对象
                return (bestMove, piData);
            }
        }

    }
}