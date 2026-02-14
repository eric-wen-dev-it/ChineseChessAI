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
    /// <summary>
    /// 训练样本：包含棋盘状态、MCTS搜索概率和最终价值
    /// </summary>
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
        /// 运行一局完整的自我对弈
        /// </summary>
        /// <param name="onMovePerformed">每步执行后的回调（用于UI同步）</param>
        /// <returns>该局对弈产生的训练数据</returns>
        public async Task<List<TrainingExample>> RunGameAsync(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            // 记录原始数据：(规范化State, 规范化Policy, 行动方)
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = 0;
            float finalResult = 0; // 0: 和棋, 1: 红胜, -1: 黑胜

            while (moveCount < 400) // 设防死循环，400步强制和棋
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        // 1. 获取当前状态的特征编码 (14x10x9)
                        // StateEncoder 内部应处理黑方视角的翻转
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

                        // 2. MCTS 搜索得到走法分布 (Pi)
                        // 使用 800 次模拟，返回 8100 维概率
                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 800);

                        // 3. 规范化 Policy 数据
                        // 如果当前是黑方，则必须将 Pi 映射到其翻转后的坐标系
                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        // 4. 选择动作执行
                        // 在前 30 步增加探索性 (Temperature Sampling)，之后选择最优步
                        Move move = (moveCount < 30) ? SelectMoveBySampling(piData) : bestMove;

                        board.Push(move.From, move.To);
                        onMovePerformed?.Invoke(board);
                        moveCount++;

                        // 5. 检查终局
                        // A. 绝杀或困毙
                        if (_generator.GenerateLegalMoves(board).Count == 0)
                        {
                            // 当前方已无路可走，判负
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        // B. 三次重复局面判定（利用之前在 Board.cs 中加的 Hash 检测）
                        // 根据规则，MoveGenerator 已经过滤了非法的长将，如果依然重复 3 次则判和
                        if (board.GetRepetitionCount() >= 3)
                        {
                            Console.WriteLine($"[通知] 检测到三次重复局面，判定和棋。");
                            finalResult = 0.0f;
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[致命] 线程对弈异常 (步数: {moveCount}): {ex.Message}");
                    throw;
                }
            }

            // 输出最终棋谱 UCCI 序列
            Console.WriteLine($"[棋谱序列] {board.GetMoveHistoryString()}");

            // 6. 将对弈历史转化为训练样本 (Z-Value 映射)
            return FinalizeData(gameHistory, finalResult);
        }

        /// <summary>
        /// 根据概率分布随机采样走法 (增加探索性)
        /// </summary>
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

        /// <summary>
        /// 将 8100 维的策略向量进行中心对称翻转 (9-r, 8-c)
        /// </summary>
        private float[] FlipPolicy(float[] originalPi)
        {
            float[] flippedPi = new float[8100];
            for (int i = 0; i < 8100; i++)
            {
                if (originalPi[i] <= 0)
                    continue;

                int from = i / 90;
                int to = i % 90;

                // 翻转坐标: r -> 9-r, c -> 8-c
                int r1_flip = 9 - (from / 9);
                int c1_flip = 8 - (from % 9);
                int r2_flip = 9 - (to / 9);
                int c2_flip = 8 - (to % 9);

                int idx_flip = (r1_flip * 9 + c1_flip) * 90 + (r2_flip * 9 + c2_flip);
                if (idx_flip >= 0 && idx_flip < 8100)
                    flippedPi[idx_flip] = originalPi[i];
            }
            return flippedPi;
        }

        /// <summary>
        /// 分配最终胜负价值 (V) 到每一个训练步骤
        /// </summary>
        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult)
        {
            var examples = new List<TrainingExample>();
            foreach (var step in history)
            {
                // Value 始终针对“当前行动方”
                // 如果结果是红胜(1)，红方回合 Value 为 1，黑方回合 Value 为 -1
                float valueForCurrentPlayer = step.isRedTurn ? finalResult : -finalResult;
                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
            }
            return examples;
        }
    }
}