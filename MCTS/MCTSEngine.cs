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

            // 保持 FP32，由 Trainer 决定是否使用 AMP (目前 Trainer 已回退到 FP32，保持一致)
            _model.to(DeviceType.CUDA);
            _model.eval();

            _batchInference = new BatchInference(_model, batchSize);
        }

        public async Task<Move> GetBestMoveAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            // 动态调整并行度，避免小模拟次数下线程开销过大
            int parallelTasks = 16;
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

                using (var scope = torch.NewDisposeScope())
                {
                    // 1. 记录当前是否为红方，因为 Encode 内部会翻转棋盘
                    bool isRed = board.IsRedTurn;

                    // 2. 编码 (StateEncoder 会自动处理 FlipBoard)
                    var inputTensor = StateEncoder.Encode(board).to(DeviceType.CUDA);

                    // 3. 推理 (得到的是 "红方视角" 的 Logits)
                    var (policyLogits, value) = await _batchInference.PredictAsync(inputTensor);

                    // 4. 【核心修复】如果是黑方，必须将 Policy 翻转回真实坐标系
                    // 这样神经网络输出的 "红方兵7进1" 才能正确对应到物理棋盘的 "黑方卒3进1"
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
            catch (Exception ex)
            {
                Console.WriteLine($"[MCTS Async Search Error] {ex.Message}");
            }
        }

        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            var tasks = Enumerable.Range(0, 8).Select(_ => Task.Run(async () => {
                for (int i = 0; i < simulations / 8; i++)
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

        // 复用翻转逻辑 (将 180 度旋转应用于 Policy 索引)
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