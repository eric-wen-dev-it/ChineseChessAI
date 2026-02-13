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
        private readonly BatchInference _batchInference; // 引入批量推理器
        private readonly double _cPuct = 1.5;

        // 构造函数中初始化批量推理器
        public MCTSEngine(CChessNet model, int batchSize = 16)
        {
            _model = model;
            _generator = new MoveGenerator();
            // 确保模型处于推理模式并使用 Float16
            _model.to(DeviceType.CUDA, ScalarType.Float16);
            _model.eval();
            _batchInference = new BatchInference(_model, batchSize);
        }

        /// <summary>
        /// 异步执行 MCTS 搜索。并行化模拟是大幅提速的关键。
        /// </summary>
        public async Task<Move> GetBestMoveAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            // 将模拟任务分组并行处理，而不是一个一个跑
            int parallelTasks = 16;
            int simsPerTask = simulations / parallelTasks;

            var tasks = new List<Task>();
            for (int t = 0; t < parallelTasks; t++)
            {
                tasks.Add(Task.Run(async () =>
                {
                    for (int i = 0; i < simsPerTask; i++)
                    {
                        Board tempBoard = CloneBoard(board);
                        await SearchAsync(root, tempBoard); // 异步搜索
                    }
                }));
            }

            await Task.WhenAll(tasks);

            if (root.Children.Count == 0)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                return legalMoves.Count == 0 ? throw new Exception("无合法走法") : legalMoves[0];
            }

            return root.Children.OrderByDescending(x => x.Value.N).First().Key;
        }

        /// <summary>
        /// 异步搜索逻辑：这是优化的核心，将 GPU 推理请求发送给 BatchInference
        /// </summary>
        private async Task SearchAsync(MCTSNode node, Board board)
        {
            try
            {
                // A. 选择阶段 (同步执行，树操作极快)
                if (!node.IsLeaf)
                {
                    if (node.Children.Count == 0)
                        return;

                    var bestChild = node.Children
                        .OrderByDescending(x => x.Value.GetPUCTValue(_cPuct, node.N))
                        .First();

                    board.Push(bestChild.Key.From, bestChild.Key.To);
                    await SearchAsync(bestChild.Value, board);
                    return;
                }

                // B. 扩展与评估阶段 (异步调用 GPU 批量推理)
                using (var scope = torch.NewDisposeScope())
                {
                    // 1. 编码局面并转换为 Float16
                    var inputTensor = StateEncoder.Encode(board).to(DeviceType.CUDA, ScalarType.Float16);

                    // 2. 调用批量推理器，此处 CPU 会释放线程去处理其他节点的选择
                    var (policyLogits, value) = await _batchInference.PredictAsync(inputTensor);

                    var legalMoves = _generator.GenerateLegalMoves(board);
                    if (legalMoves.Count == 0)
                    {
                        node.Update(-1.0);
                        return;
                    }

                    // 3. 过滤并展开
                    var filteredPolicy = GetFilteredPolicy(policyLogits, legalMoves);
                    node.Expand(filteredPolicy);

                    // C. 反向传播
                    node.Update(value);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MCTS Async Search Error] {ex.Message}");
            }
        }

        /// <summary>
        /// 针对 SelfPlay 优化的数组返回方法，同样改为异步批量化
        /// </summary>
        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            // 同样使用并行模拟提升速度
            var tasks = Enumerable.Range(0, 8).Select(_ => Task.Run(async () => {
                for (int i = 0; i < simulations / 8; i++)
                {
                    await SearchAsync(root, CloneBoard(board));
                }
            }));
            await Task.WhenAll(tasks);

            if (root.Children.Count == 0)
            {
                return (_generator.GenerateLegalMoves(board)[0], new float[8100]);
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

        private IEnumerable<(Move move, double prob)> GetFilteredPolicy(torch.Tensor logits, List<Move> legalMoves)
        {
            // 注意：此处 logits 已经是 CPU 上的 Float32 Tensor (由 BatchInference.cs 处理)
            var probs = new List<(Move, double)>();
            float maxLogit = float.MinValue;
            var validMoves = new List<(Move, float)>();

            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                {
                    float val = logits[idx].item<float>(); // 直接读取索引
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

        private Board CloneBoard(Board original)
        {
            var newBoard = new Board();
            newBoard.LoadState(original.GetState(), original.IsRedTurn);
            return newBoard;
        }
    }
}