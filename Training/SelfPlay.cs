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
        /// 执行一局完整的自我对弈并返回训练数据，支持 UI 实时回调
        /// </summary>
        /// <param name="onMovePerformed">每一步走完后的回调函数，用于更新 UI</param>
        public List<TrainingExample> RunGame(Action<Board>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            // 初始状态回调一次，显示开局棋盘
            onMovePerformed?.Invoke(board);

            var gameHistory = new List<(torch.Tensor state, torch.Tensor policy)>();
            int moveCount = 0;

            while (moveCount < 400)
            {
                var stateTensor = StateEncoder.Encode(board);

                // MCTS 搜索逻辑
                (Move bestMove, torch.Tensor pi) = _engine.GetMoveWithProbabilities(board, 800);

                // 记录数据
                gameHistory.Add((stateTensor.squeeze(0), pi));

                // 执行走子
                board.Push(bestMove.From, bestMove.To);

                // 关键修改：执行回调，通知 UI 更新棋盘
                onMovePerformed?.Invoke(board);

                moveCount++;

                if (_generator.GenerateLegalMoves(board).Count == 0)
                    break;
            }

            // 胜负判断逻辑
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