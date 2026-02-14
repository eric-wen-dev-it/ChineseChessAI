using System;
using System.Collections.Generic;
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

        public SelfPlay(MCTSEngine engine)
        {
            _engine = engine;
            _generator = new MoveGenerator();
        }

        public async Task<List<TrainingExample>> RunGameAsync(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            // 记录原始数据：(规范化Input, 原始Policy, 玩家)
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = 0;

            while (moveCount < 400)
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        // 1. 获取 State (StateEncoder 内部已经处理了 FlipBoard，所以这里拿到的是规范化后的数据)
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).data<float>().ToArray();

                        // 2. MCTS 搜索 (得到的是基于真实棋盘的绝对坐标 Policy)
                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 800);

                        // 3. 关键修复：如果当前是黑方，Input 已经被翻转了，Output (Policy) 也必须对应翻转
                        // 否则神经网络会学习到“看着红方的棋盘，却走出黑方的坐标”，导致逻辑错乱
                        if (!isRed)
                        {
                            piData = FlipPolicy(piData);
                        }

                        gameHistory.Add((stateData, piData, isRed));

                        board.Push(bestMove.From, bestMove.To);
                        onMovePerformed?.Invoke(board);
                        moveCount++;

                        if (_generator.GenerateLegalMoves(board).Count == 0)
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[致命] 线程对弈异常 (步数: {moveCount}): {ex.Message}");
                    throw;
                }
            }

            // 最终胜负 (1: 红胜, -1: 黑胜)
            float finalResult = board.IsRedTurn ? -1.0f : 1.0f;
            return FinalizeData(gameHistory, finalResult);
        }

        /// <summary>
        /// 将 8100 维的策略向量进行中心对称翻转 (用于黑方视角规范化)
        /// </summary>
        private float[] FlipPolicy(float[] originalPi)
        {
            float[] flippedPi = new float[8100];

            // 遍历所有可能的动作索引
            for (int i = 0; i < 8100; i++)
            {
                if (originalPi[i] == 0)
                    continue;

                // 解码索引 -> 坐标
                int from = i / 90;
                int to = i % 90;

                int r1 = from / 9;
                int c1 = from % 9;
                int r2 = to / 9;
                int c2 = to % 9;

                // 执行中心对称翻转 (9-r, 8-c)
                int r1_flip = 9 - r1;
                int c1_flip = 8 - c1;
                int r2_flip = 9 - r2;
                int c2_flip = 8 - c2;

                // 重新编码为索引
                int from_flip = r1_flip * 9 + c1_flip;
                int to_flip = r2_flip * 9 + c2_flip;
                int idx_flip = from_flip * 90 + to_flip;

                flippedPi[idx_flip] = originalPi[i];
            }
            return flippedPi;
        }

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult)
        {
            var examples = new List<TrainingExample>();
            foreach (var step in history)
            {
                // AlphaZero 标准：Value 始终是针对“当前行动方”的收益
                // 如果最终结果是 红胜(1)，而当前步是红方(isRedTurn=true)，则 Value = 1
                // 如果最终结果是 红胜(1)，而当前步是黑方(isRedTurn=false)，则 Value = -1
                float valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;

                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }
    }
}