using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using TorchSharp;

namespace ChineseChessAI.Training
{
    public record TrainingExample(float[] State, float[] Policy, float Value);

    public record GameResult(List<TrainingExample> Examples, string EndReason, string ResultStr, int MoveCount);

    public class SelfPlay
    {
        private readonly MCTSEngine _engine;
        private readonly MoveGenerator _generator;
        private readonly Random _random = new Random();

        public SelfPlay(MCTSEngine engine)
        {
            _engine = engine;
            _generator = new MoveGenerator();
        }

        public async Task<GameResult> RunGameAsync(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = 0;
            float finalResult = 0;
            string endReason = "进行中";

            while (true)
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

                        // 核心：使用 3200 次模拟量进行深度搜索
                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 3200);

                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        // 【重大调整】优化温度系数逻辑
                        // 前 30 步 (约 15 回合) 保持 1.0 的高探索度，确保开局库多样性
                        // 30 步之后立即降至 0.01 (趋近于只选概率最高的步子)，保证中残局样本的极高棋力
                        double temperature = (moveCount < 60) ? 1.0 : 0.01;

                        Move move = SelectMoveByTemperature(piData, temperature);

                        board.Push(move.From, move.To);
                        onMovePerformed?.Invoke(board);
                        moveCount++;

                        // --- 终局判定 ---
                        var legalMoves = _generator.GenerateLegalMoves(board);
                        if (legalMoves.Count == 0)
                        {
                            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
                            endReason = inCheck ? "绝杀" : "困毙";
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        // 8次重复判定：配合低温度更容易检测出死结
                        if (board.GetRepetitionCount() >= 8)
                        {
                            endReason = "八次重复局面";
                            finalResult = 0.0f;
                            break;
                        }

                        // 步数硬上限
                        if (moveCount >= 300)
                        {
                            endReason = "步数达到 1000 步限制";
                            finalResult = 0.0f;
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[致命] 对弈异常 (步数: {moveCount}): {ex.Message}");
                    throw;
                }
            }

            string resultStr = finalResult == 0 ? "平局" : (finalResult > 0 ? "红胜" : "黑胜");
            return new GameResult(FinalizeData(gameHistory, finalResult), endReason, resultStr, moveCount);
        }

        // 温度采样：P_new = P_old ^ (1/T)
        private Move SelectMoveByTemperature(float[] piData, double temperature)
        {
            // 当温度极低时，直接取概率最大的走法，避免浮点溢出
            if (temperature < 0.1)
            {
                int bestIdx = 0;
                float maxProb = -1f;
                for (int i = 0; i < piData.Length; i++)
                {
                    if (piData[i] > maxProb)
                    {
                        maxProb = piData[i];
                        bestIdx = i;
                    }
                }
                return Move.FromNetworkIndex(bestIdx);
            }

            double[] poweredPi = piData.Select(p => Math.Pow(p, 1.0 / temperature)).ToArray();
            double sum = poweredPi.Sum();

            double r = _random.NextDouble() * sum;
            double cumulative = 0;
            for (int i = 0; i < poweredPi.Length; i++)
            {
                cumulative += poweredPi[i];
                if (r <= cumulative)
                    return Move.FromNetworkIndex(i);
            }
            return Move.FromNetworkIndex(Array.IndexOf(piData, piData.Max()));
        }

        private float[] FlipPolicy(float[] originalPi)
        {
            float[] flippedPi = new float[8100];
            for (int i = 0; i < 8100; i++)
            {
                if (originalPi[i] <= 0)
                    continue;
                int from = i / 90, to = i % 90;
                int r1_f = 9 - (from / 9), c1_f = 8 - (from % 9);
                int r2_f = 9 - (to / 9), c2_f = 8 - (to % 9);
                int idx_f = (r1_f * 9 + c1_f) * 90 + (r2_f * 9 + c2_f);
                if (idx_f >= 0 && idx_f < 8100)
                    flippedPi[idx_f] = originalPi[i];
            }
            return flippedPi;
        }

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult)
        {
            var examples = new List<TrainingExample>();
            // 将平局惩罚从 -0.9 调整为更敏感的 -0.5
            float drawPenalty = -0.5f;

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                float valueForCurrentPlayer;

                if (Math.Abs(finalResult) < 0.001f) // 处理平局
                {
                    // 只要是平局，所有步数都给予一定的负面评价，
                    // 越接近结尾的步数惩罚越重（说明它没能打破僵局）
                    float progression = (float)i / history.Count;
                    valueForCurrentPlayer = drawPenalty * progression;
                }
                else
                {
                    valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;
                }

                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }
    }
}