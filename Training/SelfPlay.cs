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

        /// <summary>
        /// 异步运行单局自我对弈。
        /// 配合异步 MCTSEngine，能够释放 CPU 线程以支持更高并发。
        /// </summary>
        public async Task<List<TrainingExample>> RunGameAsync(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            var gameHistory = new List<(float[] state, float[] policy)>();
            int moveCount = 0;

            // 限制最大步数防止死循环
            while (moveCount < 400)
            {
                try
                {
                    // 使用异步作用域管理内存
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        // 1. 获取当前盘面编码数组
                        // 直接通过 Encode 获取 Tensor 并转为数组存储，用于后续训练
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).data<float>().ToArray();

                        // 2. 异步调用 MCTS 引擎获取走法和概率分布 (π)
                        // 这里使用的是之前优化过的异步批量推理方法
                        (Move bestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 800);

                        // 3. 记录历史数据
                        gameHistory.Add((stateData, piData));

                        // 4. 执行走法
                        board.Push(bestMove.From, bestMove.To);

                        // 5. 触发 UI 更新回调（如果有）
                        onMovePerformed?.Invoke(board);
                        moveCount++;

                        // 6. 检查胜负（无合法走法即为困毙/绝杀）
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

            // 根据游戏结束时的回合方确定胜负结果 (1: 红胜, -1: 黑胜)
            float gameResult = board.IsRedTurn ? -1.0f : 1.0f;
            return FinalizeData(gameHistory, gameResult);
        }

        /// <summary>
        /// 整理训练数据，根据每一步的回合方转换胜负视角。
        /// </summary>
        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy)> history, float result)
        {
            var examples = new List<TrainingExample>();
            for (int i = 0; i < history.Count; i++)
            {
                // 如果是第 0, 2, 4... 步（红方走），结果保持 result
                // 如果是第 1, 3, 5... 步（黑方走），结果取反 -result
                float perspectiveResult = (i % 2 == 0) ? result : -result;
                examples.Add(new TrainingExample(history[i].state, history[i].policy, perspectiveResult));
            }
            return examples;
        }
    }
}