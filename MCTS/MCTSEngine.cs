using ChineseChessAI.Core;
using ChineseChessAI.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.MCTS
{
    public class MCTSEngine
    {
        private readonly CChessNet _model;
        private readonly MoveGenerator _generator;
        private readonly BatchInference _batchInference;
        private readonly double _cPuct = 1.5;

        public MCTSEngine(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _generator = new MoveGenerator();

            if (torch.cuda.is_available())
                _model.to(DeviceType.CUDA);
            _model.eval();

            _batchInference = new BatchInference(_model, batchSize);
        }

        public async Task<Move> GetBestMoveAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            // 保持较低的并发度，稳定性优先
            int parallelTasks = 4;
            int simsPerTask = simulations / parallelTasks;
            if (simsPerTask < 1)
            {
                parallelTasks = simulations;
                simsPerTask = 1;
            }

            var tasks = new List<Task>();
            for (int t = 0; t < parallelTasks; t++)
            {
                tasks.Add(Task.Run(async () =>
                {
                    for (int i = 0; i < simsPerTask; i++)
                    {
                        Board tempBoard = CloneBoard(board);
                        await SearchAsync(root, tempBoard);
                    }
                }));
            }

            await Task.WhenAll(tasks);

            if (root.Children.IsEmpty)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                return legalMoves.Count == 0 ? throw new Exception("无合法走法") : legalMoves[0];
            }

            return root.Children.OrderByDescending(x => x.Value.N).First().Key;
        }

        private async Task SearchAsync(MCTSNode node, Board board)
        {
            try
            {
                if (!node.IsLeaf)
                {
                    if (node.Children.IsEmpty)
                        return;

                    var bestChild = node.Children
                        .OrderByDescending(x => x.Value.GetPUCTValue(_cPuct, node.N))
                        .First();

                    board.Push(bestChild.Key.From, bestChild.Key.To);
                    await SearchAsync(bestChild.Value, board);
                    return;
                }

                // 准备数据变量
                float[] inputData;
                bool isRed = board.IsRedTurn;

                // 【关键修复】使用局部 Scope 快速生成数据并转换为 Array，随后立即销毁 Tensor
                // 这样 Tensor 永远不会跨越 await 边界，也不会跨线程
                using (var scope = torch.NewDisposeScope())
                {
                    var tensor = StateEncoder.Encode(board);

                    // 确保数据在 CPU 且为 float32
                    if (tensor.device_type != DeviceType.CPU)
                        tensor = tensor.cpu();
                    tensor = tensor.to_type(ScalarType.Float32);

                    inputData = tensor.data<float>().ToArray();
                } // Tensor 在此处被物理销毁，非常安全

                // 发送纯数组进行推理
                var (policyLogits, value) = await _batchInference.PredictAsync(inputData);

                // 后处理
                if (!isRed)
                {
                    policyLogits = FlipPolicy(policyLogits);
                }

                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count == 0)
                {
                    node.Update(-1.0);
                    return;
                }

                var filteredPolicy = GetFilteredPolicy(policyLogits, legalMoves);
                node.Expand(filteredPolicy);
                node.Update(value);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MCTS Async Search Error] {ex.Message}");
            }
        }

        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);
            int numThreads = 4;
            int baseSims = simulations / numThreads;
            int extraSims = simulations % numThreads;

            var tasks = Enumerable.Range(0, numThreads).Select(t => Task.Run(async () =>
            {
                // 确保总数准确，将余数分配给第一个线程
                int taskSims = (t == 0) ? baseSims + extraSims : baseSims;

                for (int i = 0; i < taskSims; i++)
                {
                    // SearchAsync 内部应保持对 root 节点的并发安全操作
                    await SearchAsync(root, CloneBoard(board));
                }
            }));

            await Task.WhenAll(tasks);

            // 检查是否有合法走法
            if (root.Children.IsEmpty)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count == 0)
                    throw new Exception("无合法走法");
                return (legalMoves[0], new float[8100]);
            }

            // 计算概率分布 (Pi)
            float[] piData = new float[8100];
            // 使用 Sum() 统计所有子节点的访问次数总和
            double totalVisits = root.Children.Values.Sum(x => x.N);

            foreach (var child in root.Children)
            {
                int moveIdx = child.Key.ToNetworkIndex();
                if (moveIdx >= 0 && moveIdx < 8100)
                {
                    // 归一化访问次数：pi = N_i / Total_N
                    piData[moveIdx] = (float)(child.Value.N / (totalVisits > 0 ? totalVisits : 1));
                }
            }

            // 选择访问次数最多的走法作为最佳走法
            var bestMove = root.Children.OrderByDescending(x => x.Value.N).First().Key;
            return (bestMove, piData);
        }

        // ... GetFilteredPolicy, FlipPolicy, CloneBoard 保持不变 ...
        private IEnumerable<(Move move, double prob)> GetFilteredPolicy(float[] logits, List<Move> legalMoves)
        {
            var probs = new List<(Move, double)>();
            float maxLogit = float.MinValue;
            var validMoves = new List<(Move, float)>();

            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                {
                    float val = logits[idx];
                    if (val > maxLogit)
                        maxLogit = val;
                    validMoves.Add((move, val));
                }
            }

            double sum = 0;
            foreach (var (move, val) in validMoves)
            {
                double p = Math.Exp(val - maxLogit);
                probs.Add((move, p));
                sum += p;
            }

            return probs.Select(x => (x.Item1, x.Item2 / (sum > 0 ? sum : 1)));
        }

        private float[] FlipPolicy(float[] originalPi)
        {
            float[] flippedPi = new float[8100];
            for (int i = 0; i < 8100; i++)
            {
                if (originalPi[i] == 0)
                    continue;
                int from = i / 90;
                int to = i % 90;
                int r1 = from / 9;
                int c1 = from % 9;
                int r2 = to / 9;
                int c2 = to % 9;
                int r1_flip = 9 - r1;
                int c1_flip = 8 - c1;
                int r2_flip = 9 - r2;
                int c2_flip = 8 - c2;
                int from_flip = r1_flip * 9 + c1_flip;
                int to_flip = r2_flip * 9 + c2_flip;
                int idx_flip = from_flip * 90 + to_flip;
                if (idx_flip >= 0 && idx_flip < 8100)
                    flippedPi[idx_flip] = originalPi[i];
            }
            return flippedPi;
        }

        private Board CloneBoard(Board original)
        {
            var newBoard = new Board();
            newBoard.LoadState(original.GetState(), original.IsRedTurn);
            return newBoard;
        }
    }
}