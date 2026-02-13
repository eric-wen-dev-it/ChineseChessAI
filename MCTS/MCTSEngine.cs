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

            // 优化 1: 移除强制 Float16，使用默认 Float32
            // 具体的混合精度计算将由 Trainer 和 BatchInference 中的 AMP (autocast) 接管
            _model.to(DeviceType.CUDA);
            _model.eval();

            _batchInference = new BatchInference(_model, batchSize);
        }

        /// <summary>
        /// 异步执行 MCTS 搜索。并行化模拟是大幅提速的关键。
        /// </summary>
        public async Task<Move> GetBestMoveAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            // 将模拟任务分组并行处理
            int parallelTasks = 16;
            int simsPerTask = simulations / parallelTasks;
            if (simsPerTask == 0)
                simsPerTask = 1;

            var tasks = new List<Task>();
            for (int t = 0; t < parallelTasks; t++)
            {
                tasks.Add(Task.Run(async () =>
                {
                    for (int i = 0; i < simsPerTask; i++)
                    {
                        // 必须克隆 Board，因为 MCTS 在不同线程中会修改棋盘状态
                        Board tempBoard = CloneBoard(board);
                        await SearchAsync(root, tempBoard);
                    }
                }));
            }

            await Task.WhenAll(tasks);

            // 兼容 ConcurrentDictionary 的判空方式 (IsEmpty 性能优于 Count)
            if (root.Children.IsEmpty)
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
                    if (node.Children.IsEmpty)
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
                    // 优化 2: 保持 FP32 输入，不要手动转 Float16
                    var inputTensor = StateEncoder.Encode(board).to(DeviceType.CUDA);

                    // 调用批量推理器，直接获取 float[] 数组，无需处理 Tensor 句柄
                    var (policyLogits, value) = await _batchInference.PredictAsync(inputTensor);

                    var legalMoves = _generator.GenerateLegalMoves(board);
                    if (legalMoves.Count == 0)
                    {
                        node.Update(-1.0);
                        return;
                    }

                    // 优化 3: 传递 float[] 进行策略过滤
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

            // 并行模拟
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

        // 优化 4: 参数改为 float[] logits，直接操作数组提升性能
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
                    float val = logits[idx]; // 直接数组访问，比 Tensor.item<float> 快得多
                    if (val > maxLogit)
                        maxLogit = val;
                    validMoves.Add((move, val));
                }
            }

            // Stable Softmax
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