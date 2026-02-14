using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;

namespace ChineseChessAI.Training
{
    public record TrainingExample(float[] State, float[] Policy, float Value);

    // 【新增】封装对弈结果，便于 UI 获取信息
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

        // 【修改】返回类型改为 Task<GameResult>
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

                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 800);

                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        Move move = (moveCount < 30) ? SelectMoveBySampling(piData) : bestMove;

                        board.Push(move.From, move.To);
                        onMovePerformed?.Invoke(board);
                        moveCount++;

                        // --- 终局判定与原因细化 ---
                        var legalMoves = _generator.GenerateLegalMoves(board);
                        if (legalMoves.Count == 0)
                        {
                            // 区分绝杀（在将军状态下无棋可走）和困毙（非将军状态下无棋可走）
                            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
                            endReason = inCheck ? "绝杀" : "困毙";
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        if (board.GetRepetitionCount() >= 3)
                        {
                            endReason = "三次重复局面";
                            finalResult = 0.0f;
                            break;
                        }

                        if (moveCount >= 500)
                        {
                            endReason = "步数达到 500 步限制";
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

            // 返回包含所有信息的 GameResult
            return new GameResult(FinalizeData(gameHistory, finalResult), endReason, resultStr, moveCount);
        }

        private Move SelectMoveBySampling(float[] piData)
        {
            double r = _random.NextDouble();
            double cumulative = 0;
            for (int i = 0; i < piData.Length; i++)
            {
                cumulative += piData[i];
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

        // 位置：Training/SelfPlay.cs 类内部

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult)
        {
            var examples = new List<TrainingExample>();

            // 【关键修改】设置和棋惩罚值
            // 推荐值：-0.05 到 -0.1
            // 逻辑：如果 AI 发现这一步走完后预估价值是 -0.1 (和棋)，而另一步是 -0.5 (输棋)，它会选和棋。
            // 但如果它发现有一步是 0.1 (微弱优势)，它就会放弃和棋去拼一把。
            float drawPenalty = -0.1f;

            foreach (var step in history)
            {
                float valueForCurrentPlayer;

                // 判定：如果 finalResult 为 0，说明是和棋（三次重复或步数耗尽）
                if (Math.Abs(finalResult) < 0.001f)
                {
                    // 无论当前是红方还是黑方，和棋都记为负分
                    valueForCurrentPlayer = drawPenalty;
                }
                else
                {
                    // 有胜负：红胜(1) 或 黑胜(-1)
                    // 逻辑不变：如果当前是红方且红胜，得1分；如果当前是黑方且红胜，得-1分
                    valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;
                }

                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }
    }
}