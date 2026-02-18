using ChineseChessAI.Core;
using ChineseChessAI.NeuralNetwork;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.MCTS
{
    public class MCTSEngine
    {
        private readonly CChessNet _model;
        private readonly MoveGenerator _generator;
        private readonly BatchInference _batchInference;
        private readonly double _cPuct = 2.0;

        public MCTSEngine(CChessNet model, int batchSize = 64)
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
            int parallelTasks = 4;
            int simsPerTask = simulations / parallelTasks;

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
                    board.Pop();
                    return;
                }

                float[] inputData;
                bool isRed = board.IsRedTurn;

                using (var scope = torch.NewDisposeScope())
                {
                    var tensor = StateEncoder.Encode(board);
                    if (tensor.device_type != DeviceType.CPU)
                        tensor = tensor.cpu();
                    tensor = tensor.to_type(ScalarType.Float32);
                    inputData = tensor.data<float>().ToArray();
                }

                var (policyLogits, value) = await _batchInference.PredictAsync(inputData);

                if (!isRed)
                    policyLogits = FlipPolicy(policyLogits);

                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count == 0)
                {
                    node.Update(-1.0);
                    return;
                }

                // --- 核心优化：在搜索阶段干预重复局面 ---
                var rawPolicy = GetFilteredPolicy(policyLogits, legalMoves).ToList();
                double adjustedValue = value;

                for (int i = 0; i < rawPolicy.Count; i++)
                {
                    var move = rawPolicy[i].move;

                    // 利用 Zobrist 预判下一步是否会导致三复
                    if (board.WillCauseThreefoldRepetition(move.From, move.To))
                    {
                        // 1. 降低选择概率
                        rawPolicy[i] = (move, rawPolicy[i].prob * 0.0001);
                        // 2. 惩罚分值回传：认为这条路径价值极低
                        adjustedValue = -0.9;
                    }
                    // 如果已经开始重复（哪怕只是二次重复），也轻微惩罚以鼓励变招
                    else if (board.GetRepetitionCount() >= 2)
                    {
                        rawPolicy[i] = (move, rawPolicy[i].prob * 0.1);
                    }
                }

                node.Expand(rawPolicy);
                node.Update(adjustedValue);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MCTS Search Error] {ex.Message}");
            }
        }

        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);
            int numThreads = 24;
            int baseSims = simulations / numThreads;
            int extraSims = simulations % numThreads;

            var tasks = Enumerable.Range(0, numThreads).Select(t => Task.Run(async () =>
            {
                int taskSims = (t == 0) ? baseSims + extraSims : baseSims;
                for (int i = 0; i < taskSims; i++)
                {
                    await SearchAsync(root, CloneBoard(board));
                }
            }));

            await Task.WhenAll(tasks);

            if (root.Children.IsEmpty)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count == 0)
                    throw new Exception("无合法走法");
                return (legalMoves[0], new float[8100]);
            }

            float[] piData = new float[8100];
            double totalVisits = root.Children.Values.Sum(x => x.N);

            foreach (var child in root.Children)
            {
                int moveIdx = child.Key.ToNetworkIndex();
                if (moveIdx >= 0 && moveIdx < 8100)
                {
                    piData[moveIdx] = (float)(child.Value.N / (totalVisits > 0 ? totalVisits : 1));
                }
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
                int from = i / 90, to = i % 90;
                int r1 = from / 9, c1 = from % 9, r2 = to / 9, c2 = to % 9;
                int from_f = (9 - r1) * 9 + (8 - c1);
                int to_f = (9 - r2) * 9 + (8 - c2);
                int idx_f = from_f * 90 + to_f;
                if (idx_f >= 0 && idx_f < 8100)
                    flippedPi[idx_f] = originalPi[i];
            }
            return flippedPi;
        }

        private Board CloneBoard(Board original)
        {
            var newBoard = new Board();
            // 1. 物理局面克隆
            newBoard.LoadState(original.GetState(), original.IsRedTurn);

            // 2. 历史记录深度同步（非常重要，用于长将长捉检测）
            // 注意：Stack 是后进先出，Reverse() 保证按时间顺序重新压入
            foreach (var state in original.GetHistory().Reverse())
            {
                newBoard.RecordHistory(state);
            }
            return newBoard;
        }
    }
}