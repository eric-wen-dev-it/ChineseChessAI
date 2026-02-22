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

        /// <summary>
        /// 执行一局自对弈过程
        /// </summary>
        /// <param name="onMovePerformed">异步回调，用于 UI 演示动画</param>
        public async Task<GameResult> RunGameAsync(Func<Board, Task>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            // --- 1. 随机开局：前 4 步完全随机，确保每一局阵型不同 ---
            for (int i = 0; i < 4; i++)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count > 0)
                {
                    var randomMove = legalMoves[_random.Next(legalMoves.Count)];
                    board.Push(randomMove.From, randomMove.To);
                    if (onMovePerformed != null)
                        await onMovePerformed.Invoke(board);
                }
            }

            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = 4; // 从第 5 步正式开始
            float finalResult = 0;
            string endReason = "进行中";

            // --- 2. 正式对弈循环 ---
            while (true)
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        // 编码当前局面
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

                        // MCTS 搜索获取走法概率分布 (3200次模拟)
                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 3200);

                        // 记录训练数据（黑方视角需翻转）
                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        // --- 核心优化：动态调整探索温度 ---
                        // 150步之前保持 1.0 的高随机性
                        // 150步之后降至 0.4 而非 0.01，鼓励在残局也寻找变招
                        double temperature = (moveCount < 150) ? 1.0 : 0.4;

                        Move move = SelectMoveByTemperature(piData, temperature);

                        // 执行走子
                        board.Push(move.From, move.To);

                        // 触发 UI 动画回调
                        if (onMovePerformed != null)
                            await onMovePerformed.Invoke(board);

                        moveCount++;

                        // --- 3. 终局判定 ---
                        var legalMoves = _generator.GenerateLegalMoves(board);

                        // A. 绝杀或困毙判定
                        if (legalMoves.Count == 0)
                        {
                            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
                            endReason = inCheck ? "绝杀" : "困毙";
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        // B. 平局判定 (长将重复 8 次或达到 600 步限制)
                        if (board.GetRepetitionCount() >= 8 || moveCount >= 600)
                        {
                            endReason = moveCount >= 600 ? "步数限制" : "八次重复";
                            finalResult = 0.0f; // 初始设为 0，在 FinalizeData 中根据子力细化
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[SelfPlay Error] {ex.Message}");
                    break;
                }
            }

            string resultStr = finalResult == 0 ? "平局" : (finalResult > 0 ? "红胜" : "黑胜");

            // 构造带子力分评估的训练样本
            return new GameResult(FinalizeData(gameHistory, finalResult, board), endReason, resultStr, moveCount);
        }

        /// <summary>
        /// 基于温度系数选择走法：P_new = P_old ^ (1/T)
        /// </summary>
        private Move SelectMoveByTemperature(float[] piData, double temperature)
        {
            if (temperature < 0.1)
            {
                return Move.FromNetworkIndex(Array.IndexOf(piData, piData.Max()));
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

        /// <summary>
        /// 翻转策略网络输出（用于黑方视角对齐）
        /// </summary>
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

        /// <summary>
        /// 完善样本价值评估，引入子力占优惩罚/奖励
        /// </summary>
        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult, Board finalBoard)
        {
            var examples = new List<TrainingExample>();
            float adjustedResult = finalResult;

            // --- 子力评估逻辑：打破平局时的零反馈 ---
            if (Math.Abs(finalResult) < 0.001f)
            {
                float redMaterial = CalculateMaterialScore(finalBoard, true);
                float blackMaterial = CalculateMaterialScore(finalBoard, false);

                if (redMaterial > blackMaterial)
                    adjustedResult = 0.15f; // 红方子力占优，给小正分
                else if (blackMaterial > redMaterial)
                    adjustedResult = -0.15f; // 黑方子力占优
                else
                    adjustedResult = -0.1f; // 纯平局轻微惩罚，鼓励分出胜负
            }

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                // 当前步的价值：若是当前选手的回合，则直接使用结果，否则反转
                float valueForCurrentPlayer = step.isRedTurn ? adjustedResult : -adjustedResult;
                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }

        /// <summary>
        /// 计算局面子力分值
        /// 帅:0, 仕:2, 相:2, 马:4, 车:9, 炮:4.5, 兵:1
        /// </summary>
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
                        1 => 0,    // 帅本身不计分
                        2 => 2,    // 仕
                        3 => 2,    // 相
                        4 => 4,    // 马
                        5 => 9,    // 车
                        6 => 4.5f, // 炮
                        7 => 1,    // 兵
                        _ => 0
                    };
                }
            }
            return score;
        }
    }
}