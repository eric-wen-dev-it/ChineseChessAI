using System;
using System.Collections.Generic;
using TorchSharp;
using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;

namespace ChineseChessAI.Training
{
    // TrainingExample 保持数组形式是正确的
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
            onMovePerformed?.Invoke(board);

            var gameHistory = new List<(torch.Tensor state, torch.Tensor policy)>();
            int moveCount = 0;

            while (moveCount < 400)
            {
                // 注意：这里产生的 Tensor 仍在 DisposeScope 管理下
                var stateTensor = StateEncoder.Encode(board);

                (Move bestMove, torch.Tensor pi) = _engine.GetMoveWithProbabilities(board, 800);

                // 记录数据时，使用 squeeze 确保维度正确
                gameHistory.Add((stateTensor.squeeze(0), pi));

                board.Push(bestMove.From, bestMove.To);
                onMovePerformed?.Invoke(board);

                moveCount++;

                if (_generator.GenerateLegalMoves(board).Count == 0)
                    break;
            }

            float gameResult = board.IsRedTurn ? -1.0f : 1.0f;
            return FinalizeData(gameHistory, gameResult);
        }

        private List<TrainingExample> FinalizeData(List<(torch.Tensor state, torch.Tensor policy)> history, float result)
        {
            var examples = new List<TrainingExample>();

            for (int i = 0; i < history.Count; i++)
            {
                // 1. 获取当前步的元组数据
                var (s, p) = history[i];

                // 2. 根据回合反转胜负值
                float perspectiveResult = (i % 2 == 0) ? result : -result;

                // 3. 【核心修正】将 Tensor 转换为 float[] 数组
                // 使用你实际定义的变量名 s 和 p，而不是示例占位符
                var example = new TrainingExample(
                    s.detach().cpu().data<float>().ToArray(),
                    p.detach().cpu().data<float>().ToArray(),
                    perspectiveResult // 使用计算后的 perspectiveResult
                );

                examples.Add(example);
            }
            return examples;
        }
    }
}