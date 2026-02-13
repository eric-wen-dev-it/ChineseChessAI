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

        // 文件：SelfPlay.cs

        public List<TrainingExample> RunGame(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            // 关键点 1：这里的列表直接存储 float[] 数组，而不是 Tensor
            var gameHistory = new List<(float[] state, float[] policy)>();
            int moveCount = 0;

            while (moveCount < 400)
            {
                // 关键点 2：每一步都建立独立作用域，防止显存堆积
                using (var moveScope = torch.NewDisposeScope())
                {
                    var stateTensor = StateEncoder.Encode(board);
                    (Move bestMove, torch.Tensor pi) = _engine.GetMoveWithProbabilities(board, 800);

                    // 关键点 3：立即转换！断开与 Tensor 句柄的所有联系
                    gameHistory.Add((
                        stateTensor.detach().cpu().data<float>().ToArray(),
                        pi.detach().cpu().data<float>().ToArray()
                    ));

                    board.Push(bestMove.From, bestMove.To);
                    onMovePerformed?.Invoke(board);
                    moveCount++;

                    if (_generator.GenerateLegalMoves(board).Count == 0)
                        break;
                }
            }

            float gameResult = board.IsRedTurn ? -1.0f : 1.0f;

            // 关键点 4：此时 FinalizeData 接收的是数组，非常安全
            return FinalizeData(gameHistory, gameResult);
        }

        // 对应修改 FinalizeData 的参数类型
        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy)> history, float result)
        {
            var examples = new List<TrainingExample>();
            for (int i = 0; i < history.Count; i++)
            {
                float perspectiveResult = (i % 2 == 0) ? result : -result;
                // 直接创建，不再需要 detach()，因为已经是数组了
                examples.Add(new TrainingExample(history[i].state, history[i].policy, perspectiveResult));
            }
            return examples;
        }

    }
}