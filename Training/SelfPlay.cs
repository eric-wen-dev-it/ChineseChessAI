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

                        // 保持 200 次模拟，兼顾速度与深度
                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 200);

                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        // 【优化 1】引入温度系数打破死循环
                        // 前 80 步完全随机探索，80 步后保持 0.5 的温度（不完全走 bestMove）
                        double temperature = (moveCount < 80) ? 1.0 : 0.5;
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

                        // 维持 8 次重复判定，配合温度采样更容易跳出死结
                        if (board.GetRepetitionCount() >= 8)
                        {
                            endReason = "八次重复局面";
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
            return new GameResult(FinalizeData(gameHistory, finalResult), endReason, resultStr, moveCount);
        }

        // 【优化 2】新增温度采样方法
        private Move SelectMoveByTemperature(float[] piData, double temperature)
        {
            if (temperature < 0.1)
            {
                return Move.FromNetworkIndex(Array.IndexOf(piData, piData.Max()));
            }

            // P_new = P_old ^ (1/temperature)
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

        // 【优化 3】引入近因衰减的和棋惩罚
        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult)
        {
            var examples = new List<TrainingExample>();
            // 将平局基础惩罚设为极重的 -0.9
            float drawPenalty = -0.9f;

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                float valueForCurrentPlayer;

                if (Math.Abs(finalResult) < 0.001f)
                {
                    // 衰减逻辑：越靠近棋局结束（重复发生时）的步数，惩罚越重
                    // 这样可以保护开局和早期的正确知识不被过度惩罚
                    float decay = (float)(i + 1) / history.Count;
                    valueForCurrentPlayer = drawPenalty * decay;
                }
                else
                {
                    // 有胜负：1.0 或 -1.0
                    valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;
                }

                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }
    }
}