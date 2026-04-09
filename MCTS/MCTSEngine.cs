using ChineseChessAI.Core;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Training;
using TorchSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace ChineseChessAI.MCTS
{
    // 【修复】：继承 IDisposable
    public class MCTSEngine : IDisposable
    {
        private readonly CChessNet _model;
        private readonly ChineseChessRuleEngine _rules;
        private readonly BatchInference _batchInference;
        private readonly double _cPuct;

        public MCTSEngine(CChessNet model, int batchSize = 64, double cPuct = 2.5)
        {
            _model = model;
            _cPuct = cPuct;
            _rules = new ChineseChessRuleEngine();

            // 【关键修复】：不再调用 _model.to(DeviceType.CUDA)。
            // CChessNet 构造函数和 PersistentAgent 已经确保模型在 CUDA 上。
            // TorchSharp 0.105.x 的 Module._to() 对每个参数调用 param.to(device)，
            // 若返回新 Tensor 对象则 Dispose 旧对象，导致 Trainer.Adam 持有的参数引用
            // handle = IntPtr.Zero，下一次 zero_grad() 即抛 "Tensor invalid -- empty handle"。
            _model.eval();

            _batchInference = new BatchInference(_model, batchSize);
        }

        public async Task<(Move move, float[] pi)> GetMoveWithProbabilitiesAsArrayAsync(Board board, int simulations, int currentMoves = 0, int maxMoves = 999, System.Threading.CancellationToken cancellationToken = default)
        {
            var root = new MCTSNode(null, 1.0);
            await SearchAsync(root, CloneBoard(board), currentMoves, maxMoves, 0, cancellationToken);
            
            ApplyDirichletNoise(root);

            // 【优化】：降低线程数，8个线程平衡了搜索效率和系统稳定性
            int numThreads = 8;
            int baseSims = (simulations - 1) / numThreads;
            int extraSims = (simulations - 1) % numThreads;

            var tasks = Enumerable.Range(0, numThreads).Select(t => Task.Run(async () =>
            {
                int taskSims = (t == 0) ? baseSims + extraSims : baseSims;
                for (int i = 0; i < taskSims; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    await SearchAsync(root, CloneBoard(board), currentMoves, maxMoves, 0, cancellationToken);
                }
            }));

            await Task.WhenAll(tasks);

            float[] piData = new float[8100];
            var legalMoves = _rules.GetLegalMoves(board);

            if (root.Children.IsEmpty)
            {
                if (legalMoves.Count == 0)
                    throw new Exception("无合法走法");

                throw new Exception("MCTS 搜索未能展开任何节点，可能是内部异常导致");
            }

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

        private async Task SearchAsync(MCTSNode node, Board board, int currentMoves, int maxMoves, int depth, CancellationToken cancellationToken)
        {
            try
            {
                cancellationToken.ThrowIfCancellationRequested();

                // 【核心修复 BUG-2】：感知步数上限 (Horizon Awareness)
                // 如果模拟搜索达到该局的步数上限，直接根据子力优势评估
                if (currentMoves + depth >= maxMoves)
                {
                    float advantage = TrainingOrchestrator.GetBoardAdvantage(board);
                    // MCTS 要求 Update 的值必须是相对于当前走棋方的视角
                    node.Update(board.IsRedTurn ? advantage * 0.5 : -advantage * 0.5);
                    return;
                }

                if (node.IsLeaf)
                {
                    // 【核心修复】：防止多线程竞争展开同一个叶子节点
                    if (!node.TryMarkExpanding()) return;

                    try
                    {
                        var legalMoves = _rules.GetLegalMoves(board);
                        if (legalMoves.Count == 0)
                        {
                            node.Update(-1.0);
                            return;
                        }

                        float[] inputData;
                        bool isRed = board.IsRedTurn;

                        using (var scope = torch.NewDisposeScope())
                        {
                            var tensor = StateEncoder.Encode(board);
                            inputData = tensor.to_type(ScalarType.Float32).data<float>().ToArray();
                        }

                        var (policyLogits, value) = await _batchInference.PredictAsync(inputData);
                        if (!isRed) policyLogits = StateEncoder.FlipPolicy(policyLogits);

                        var rawPolicy = GetFilteredPolicy(policyLogits, legalMoves).ToList();
                        node.Expand(rawPolicy);
                        node.Update(value);
                    }
                    finally
                    {
                        node.UnmarkExpanding();
                    }
                    return;
                }

                var bestChild = node.Children
                    .OrderByDescending(x => x.Value.GetPUCTValue(_cPuct, node.N))
                    .First();

                Interlocked.Increment(ref bestChild.Value.VirtualLoss);
                board.Push(bestChild.Key.From, bestChild.Key.To);
                
                try
                {
                    await SearchAsync(bestChild.Value, board, currentMoves, maxMoves, depth + 1, cancellationToken);
                }
                finally
                {
                    board.Pop();
                    Interlocked.Decrement(ref bestChild.Value.VirtualLoss);
                }
            }
            catch (Exception)
            {
                throw;
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
                return GammaSample(alpha + 1.0, beta) * Math.Pow(Random.Shared.NextDouble(), 1.0 / alpha);
            double d = alpha - 1.0 / 3.0;
            double c = 1.0 / Math.Sqrt(9.0 * d);
            while (true)
            {
                double x, v, u = Random.Shared.NextDouble();
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
            double u1 = Math.Max(1e-10, 1.0 - Random.Shared.NextDouble());
            double u2 = Random.Shared.NextDouble();
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

        private Board CloneBoard(Board original)
        {
            return original.Clone();
        }

        // 【修复】：实现接口方法，释放后台推理资源
        public void Dispose()
        {
            _batchInference?.Dispose();
        }
    }
}
