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

            // --- 【新增】随机开局：前 4 步完全随机，确保每一局阵型不同 ---
            for (int i = 0; i < 4; i++)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count > 0)
                {
                    var randomMove = legalMoves[_random.Next(legalMoves.Count)];
                    board.Push(randomMove.From, randomMove.To);
                    onMovePerformed?.Invoke(board);
                }
            }

            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = 4; // 从第 5 步正式开始
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

                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 3200);

                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        // --- 【核心优化】大幅放宽温度系数 ---
                        // 150步之前保持 1.0 的高随机性，150步之后降至 0.4 而非 0.01
                        // 这样即便在残局，AI 也会偶尔走出“非最优”但可能打破僵局的棋
                        double temperature = (moveCount < 150) ? 1.0 : 0.4;

                        Move move = SelectMoveByTemperature(piData, temperature);

                        board.Push(move.From, move.To);
                        onMovePerformed?.Invoke(board);
                        moveCount++;

                        // --- 终局判定 (保持 600 步限制) ---
                        var legalMoves = _generator.GenerateLegalMoves(board);
                        if (legalMoves.Count == 0)
                        {
                            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
                            endReason = inCheck ? "绝杀" : "困毙";
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        if (board.GetRepetitionCount() >= 8 || moveCount >= 600)
                        {
                            endReason = moveCount >= 600 ? "步数限制" : "八次重复";
                            finalResult = 0.0f;
                            break;
                        }
                    }
                }
                catch (Exception ex) { /* ... */ }
            }

            string resultStr = finalResult == 0 ? "平局" : (finalResult > 0 ? "红胜" : "黑胜");
            // 注意：FinalizeData 现在需要传入最终的 Board 状态来计算子力分
            return new GameResult(FinalizeData(gameHistory, finalResult, board), endReason, resultStr, moveCount);
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

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult, Board finalBoard)
        {
            var examples = new List<TrainingExample>();

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                float valueForCurrentPlayer;

                if (Math.Abs(finalResult) < 0.001f) // 处理平局
                {
                    // --- 【创新】子力评估：平局时不给 0 分，给“子力占优方”奖励 ---
                    float redMaterial = CalculateMaterialScore(finalBoard, true);
                    float blackMaterial = CalculateMaterialScore(finalBoard, false);

                    if (redMaterial > blackMaterial)
                        finalResult = 0.15f; // 红优
                    else if (blackMaterial > redMaterial)
                        finalResult = -0.15f; // 黑优
                    else
                        finalResult = -0.1f; // 绝对平局轻微惩罚

                    valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;
                }
                else
                {
                    valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;
                }

                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }

        // 简单的子力价值表：帅:0, 仕:2, 相:2, 马:4, 车:9, 炮:4.5, 兵:1 (过河+1)
        private float CalculateMaterialScore(Board board, bool isRed)
        {
            float score = 0;
            for (int i = 0; i < 90; i++)
            {
                sbyte p = board.GetPiece(i);
                if (p == 0)
                    continue;
                if ((isRed && p > 0) || (!isRed && p < 0))
                {
                    int type = Math.Abs(p);
                    score += type switch
                    {
                        1 => 0,
                        2 => 2,
                        3 => 2,
                        4 => 4,
                        5 => 9,
                        6 => 4.5f,
                        7 => 1,
                        _ => 0
                    };
                }
            }
            return score;
        }
    }
}