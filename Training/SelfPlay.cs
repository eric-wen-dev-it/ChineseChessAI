using System;
using System.Collections.Generic;
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

        public List<TrainingExample> RunGame(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            var gameHistory = new List<(float[] state, float[] policy)>();
            int moveCount = 0;

            while (moveCount < 400)
            {
                // 增加针对每一步的异常捕捉，准确定位
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        // 1. 局面编码并立即转为数组
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).data<float>().ToArray();

                        // 2. 调用修改后的引擎，直接获取 float[] 数组
                        // 注意：这里需要配合下方修改后的 MCTSEngine 使用
                        (Move bestMove, float[] piData) = _engine.GetMoveWithProbabilitiesAsArray(board, 800);

                        gameHistory.Add((stateData, piData));

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
                    throw; // 向上抛出以触发 MainWindow 的日志记录
                }
            }

            float gameResult = board.IsRedTurn ? -1.0f : 1.0f;
            return FinalizeData(gameHistory, gameResult);
        }

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy)> history, float result)
        {
            var examples = new List<TrainingExample>();
            for (int i = 0; i < history.Count; i++)
            {
                float perspectiveResult = (i % 2 == 0) ? result : -result;
                examples.Add(new TrainingExample(history[i].state, history[i].policy, perspectiveResult));
            }
            return examples;
        }
    }
}