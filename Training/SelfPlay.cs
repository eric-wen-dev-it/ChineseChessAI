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
    public record TrainingExample(
        float[] State,
        float[] Policy,
        float Value
    );

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
        /// 运行一局完整的自我对弈，直到分出胜负或判定和棋
        /// </summary>
        public async Task<List<TrainingExample>> RunGameAsync(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = 0;
            float finalResult = 0; // 0: 和棋, 1: 红胜, -1: 黑胜

            // 循环直到触发 break 结束游戏
            while (true)
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        // 1. 获取当前状态特征编码
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

                        // 2. MCTS 搜索获取走法概率分布
                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 800);

                        // 3. 规范化 Policy 数据（黑方视角翻转）
                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        // 4. 选择动作执行（前30步随机采样增加探索性）
                        Move move = (moveCount < 30) ? SelectMoveBySampling(piData) : bestMove;

                        board.Push(move.From, move.To);
                        onMovePerformed?.Invoke(board);
                        moveCount++;

                        // --- 5. 终局判定逻辑 ---

                        // A. 胜负判定：检查是否存在合法走法（绝杀或困毙）
                        if (_generator.GenerateLegalMoves(board).Count == 0)
                        {
                            // 当前轮到走棋的一方已无路可走，判负
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            Console.WriteLine($"[对弈结束] 胜负已分：{(finalResult > 0 ? "红胜" : "黑胜")}，总步数: {moveCount}");
                            break;
                        }

                        // B. 和棋判定：三次重复局面
                        // 由于 MoveGenerator 已经过滤了非法的长将/长捉，若仍出现 3 次重复，则属于合法和棋
                        if (board.GetRepetitionCount() >= 3)
                        {
                            finalResult = 0.0f;
                            Console.WriteLine($"[对弈结束] 检测到三次重复局面，判定和棋，总步数: {moveCount}");
                            break;
                        }

                        // C. 和棋判定：达到自然限着（防止早期模型死循环）
                        if (moveCount >= 500)
                        {
                            finalResult = 0.0f;
                            Console.WriteLine($"[对弈结束] 达到最大步数限制，判定和棋");
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

            // 输出中文格式棋谱（如：兵1进1）
            Console.WriteLine($"[最终棋谱] {board.GetMoveHistoryString()}");

            return FinalizeData(gameHistory, finalResult);
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

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult)
        {
            var examples = new List<TrainingExample>();
            foreach (var step in history)
            {
                // Value 始终针对当前走棋方：胜则为1，负则为-1，和则为0
                float valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;
                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }
    }
}