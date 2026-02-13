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
            var gameHistory = new List<(float[] state, float[] policy)>();
            int moveCount = 0;

            while (moveCount < 400)
            {
                // 确保子线程每一步都有完全隔离的作用域
                using (var moveScope = torch.NewDisposeScope())
                {
                    // 1. 编码当前状态
                    var stateTensor = StateEncoder.Encode(board);

                    // 2. 执行搜索
                    // 注意：如果 pi 报错 empty handle，说明 engine 内部可能已经 dispose 了它
                    var mctsResult = _engine.GetMoveWithProbabilities(board, 800);
                    Move bestMove = mctsResult.Item1;
                    torch.Tensor pi = mctsResult.Item2;

                    // 3. 【核心修复】防御性提取
                    // 在这一行，我们必须确保 pi 和 stateTensor 是活着的
                    // 如果报错依然在这一行，说明错误出在 _engine 内部
                    float[] sArray = stateTensor.detach().cpu().data<float>().ToArray();
                    float[] pArray = pi.detach().cpu().data<float>().ToArray();

                    gameHistory.Add((sArray, pArray));

                    // 4. 执行移动
                    board.Push(bestMove.From, bestMove.To);
                    onMovePerformed?.Invoke(board);
                    moveCount++;

                    if (_generator.GenerateLegalMoves(board).Count == 0)
                        break;
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
                // 计算当前走棋方的胜负分
                float perspectiveResult = (i % 2 == 0) ? result : -result;

                // 直接构造 TrainingExample，无需再调用 detach()
                examples.Add(new TrainingExample(history[i].state, history[i].policy, perspectiveResult));
            }
            return examples;
        }

    }
}