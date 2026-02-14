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

            // 确保模型在 GPU
            if (torch.cuda.is_available())
                _model.to(DeviceType.CUDA);
            _model.eval();

            _batchInference = new BatchInference(_model, batchSize);
        }

        // ... GetBestMoveAsync 保持不变 ...
        public async Task<Move> GetBestMoveAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            // 降低并发度，减少内存压力
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

                // 使用 DisposeScope 管理 CPU Tensor
                using (var scope = torch.NewDisposeScope())
                {
                    bool isRed = board.IsRedTurn;

                    // 【核心修复1】只在 CPU 上生成 Tensor，不移动到 CUDA
                    // 多线程并发向 GPU 搬运数据是崩溃的主因
                    var rawTensor = StateEncoder.Encode(board);

                    // 确保是 [1, 14, 10, 9] 形状
                    if (rawTensor.shape.Length == 3)
                        rawTensor = rawTensor.unsqueeze(0);

                    // 传递 CPU Tensor 给 BatchInference
                    var (policyLogits, value) = await _batchInference.PredictAsync(rawTensor);

                    // 4. 后处理
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
            }
            catch (IndexOutOfRangeException ior)
            {
                Console.WriteLine($"[MCTS Index Error] {ior.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MCTS Async Search Error] {ex.Message}");
            }
        }

        // ... GetMoveWithProbabilitiesAsArrayAsync, GetFilteredPolicy, FlipPolicy, CloneBoard 保持不变 ...
        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            var tasks = Enumerable.Range(0, 4).Select(_ => Task.Run(async () => { // 降低并发
                for (int i = 0; i < simulations / 4; i++)
                {
                    await SearchAsync(root, CloneBoard(board));
                }
            }));
            await Task.WhenAll(tasks);

            if (root.Children.IsEmpty)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                return (legalMoves[0], new float[8100]);
            }

            float[] piData = new float[8100];
            double totalVisits = root.Children.Sum(x => x.Value.N);

            foreach (var child in root.Children)
            {
                int moveIdx = child.Key.ToNetworkIndex();
                if (moveIdx >= 0 && moveIdx < 8100)
                    piData[moveIdx] = (float)(child.Value.N / (totalVisits > 0 ? totalVisits : 1));
            }

            var bestMove = root.Children.OrderByDescending(x => x.Value.N).First().Key;
            return (bestMove, piData);
        }

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
                if (idx_flip >= 0 && idx_flip < 8100) // 安全检查
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