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
        private readonly Random _random = new Random();

        public MCTSEngine(CChessNet model, int batchSize = 64)
        {
            _model = model;
            _generator = new MoveGenerator();

            if (torch.cuda.is_available())
                _model.to(DeviceType.CUDA);
            _model.eval();

            _batchInference = new BatchInference(_model, batchSize);
        }

        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations)
        {
            var root = new MCTSNode(null, 1.0);

            // --- 1. 初次搜索/推理以展开根节点 ---
            await SearchAsync(root, CloneBoard(board));

            // --- 2. 注入狄利克雷噪声 (仅在自对弈探索阶段) ---
            ApplyDirichletNoise(root);

            // --- 3. 执行正式的大规模并行模拟 ---
            int numThreads = 24;
            int baseSims = (simulations - 1) / numThreads;
            int extraSims = (simulations - 1) % numThreads;

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

            // 计算 Pi 概率分布
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

                // --- 1. 终局判定与合法走法获取 ---
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count == 0)
                {
                    node.Update(-1.0);
                    return;
                }

                // --- 2. 【核心新增：贪婪击杀判定】 ---
                // 如果发现能直接击杀对方将帅，立即终止搜索并判定必胜
                foreach (var move in legalMoves)
                {
                    if (_generator.CanCaptureKing(board, move))
                    {
                        // 发现可以直接吃将，赋予该动作绝对权重并停止搜索
                        var instantKillPolicy = new List<(Move move, double prob)> { (move, 1.0) };
                        node.Expand(instantKillPolicy);
                        node.Update(1.0); // 立即判定为当前搜索方的必胜局面
                        return;
                    }
                }

                // --- 3. 神经网络推理部分 ---
                float[] inputData;
                bool isRed = board.IsRedTurn;

                using (var scope = torch.NewDisposeScope())
                {
                    var tensor = StateEncoder.Encode(board);
                    if (tensor.device_type != DeviceType.CPU)
                        tensor = tensor.cpu();
                    inputData = tensor.to_type(ScalarType.Float32).data<float>().ToArray();
                }

                var (policyLogits, value) = await _batchInference.PredictAsync(inputData);
                if (!isRed)
                    policyLogits = FlipPolicy(policyLogits);

                // --- 4. 局面干预与概率过滤 ---
                var rawPolicy = GetFilteredPolicy(policyLogits, legalMoves).ToList();
                double adjustedValue = value;

                for (int i = 0; i < rawPolicy.Count; i++)
                {
                    var move = rawPolicy[i].move;
                    if (board.WillCauseThreefoldRepetition(move.From, move.To))
                    {
                        rawPolicy[i] = (move, rawPolicy[i].prob * 0.0001);
                        adjustedValue = -0.95; // 强制判负，诱导搜索树避让
                    }
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

        private void ApplyDirichletNoise(MCTSNode root)
        {
            if (root.Children.IsEmpty)
                return;

            const double epsilon = 0.25;
            const double alpha = 0.3;

            var moves = root.Children.Keys.ToList();
            var noise = SampleDirichlet(moves.Count, alpha);

            for (int i = 0; i < moves.Count; i++)
            {
                var node = root.Children[moves[i]];
                node.P = (1 - epsilon) * node.P + epsilon * noise[i];
            }
        }

        private double[] SampleDirichlet(int count, double alpha)
        {
            double[] samples = new double[count];
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                double sample = GammaSample(alpha, 1.0);
                samples[i] = sample;
                sum += sample;
            }
            for (int i = 0; i < count; i++)
                samples[i] /= sum;
            return samples;
        }

        private double GammaSample(double alpha, double beta)
        {
            if (alpha < 1.0)
                return GammaSample(alpha + 1.0, beta) * Math.Pow(_random.NextDouble(), 1.0 / alpha);
            double d = alpha - 1.0 / 3.0;
            double c = 1.0 / Math.Sqrt(9.0 * d);
            while (true)
            {
                double x, v, u = _random.NextDouble();
                do
                {
                    x = NormalSample();
                    v = 1.0 + c * x;
                } while (v <= 0);
                v = v * v * v;
                if (u < 1.0 - 0.0331 * x * x * x * x)
                    return d * v / beta;
                if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
                    return d * v / beta;
            }
        }

        private double NormalSample()
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = 1.0 - _random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
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
            newBoard.LoadState(original.GetState(), original.IsRedTurn);
            foreach (var state in original.GetHistory().Reverse())
            {
                newBoard.RecordHistory(state);
            }
            return newBoard;
        }
    }
}