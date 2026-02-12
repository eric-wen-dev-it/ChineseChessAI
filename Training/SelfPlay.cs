using System;
using System.Collections.Generic;
using TorchSharp;
using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;

namespace ChineseChessAI.Training
{
    /// <summary>
    /// 自我对弈数据记录
    /// </summary>
    public record TrainingExample(torch.Tensor State, torch.Tensor Policy, float Value);

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
        /// 执行一局完整的自我对弈并返回训练数据
        /// </summary>
        public List<TrainingExample> RunGame()
        {
            var board = new Board();
            board.Reset();

            var gameHistory = new List<(torch.Tensor state, torch.Tensor policy)>();
            int moveCount = 0;

            // 游戏主循环 (设置步数上限防止长将或和棋死循环)
            // 修改后的 SelfPlay 核心循环片段
            while (moveCount < 400)
            {
                var stateTensor = StateEncoder.Encode(board);

                // 显式接收元组返回值
                (Move bestMove, torch.Tensor pi) = _engine.GetMoveWithProbabilities(board, 800);

                // 记录数据。注意：stateTensor 已经 unsqueeze(0) 了，
                // 在存入 history 前建议保持维度一致或在 Dataset 处理时统一
                gameHistory.Add((stateTensor.squeeze(0), pi));

                board.Push(bestMove.From, bestMove.To);
                moveCount++;

                if (_generator.GenerateLegalMoves(board).Count == 0)
                    break;
            }

            // 6. 游戏结束，根据胜负回填价值 Z
            // 假设最后走棋的人赢了，或者根据最后的回合判断胜负
            float gameResult = board.IsRedTurn ? -1.0f : 1.0f;

            return FinalizeData(gameHistory, gameResult);
        }

        private List<TrainingExample> FinalizeData(List<(torch.Tensor, torch.Tensor)> history, float result)
        {
            var examples = new List<TrainingExample>();
            // 对于每一手，价值 Z 应该根据当前是谁的回合进行正负反转
            // 比如最后红方赢了，那么红方走棋的状态 Z=1，黑方走棋的状态 Z=-1
            for (int i = 0; i < history.Count; i++)
            {
                float perspectiveResult = (i % 2 == 0) ? result : -result;
                examples.Add(new TrainingExample(history[i].Item1, history[i].Item2, perspectiveResult));
            }
            return examples;
        }
    }
}