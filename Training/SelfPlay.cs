using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using TorchSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace ChineseChessAI.Training
{
    // 【修复】：删除了这里重复的、旧版的 TrainingExample 定义，统一使用 Trainer.cs 中的稀疏版本

    public record GameResult(List<TrainingExample> Examples, string EndReason, string ResultStr, int MoveCount, List<Move> MoveHistory);

    public class SelfPlay
    {
        private readonly MCTSEngine _engine;
        private readonly MoveGenerator _generator;
        private readonly Random _random = new Random();

        private readonly int _maxMoves;
        private readonly int _exploreMoves;
        private readonly float _materialBias;

        public SelfPlay(MCTSEngine engine, int maxMoves = 250, int exploreMoves = 40, float materialBias = 0.05f)
        {
            _engine = engine;
            _generator = new MoveGenerator();
            _maxMoves = maxMoves;
            _exploreMoves = exploreMoves;
            _materialBias = materialBias;
        }

        public async Task<GameResult> RunGameAsync(Func<Board, Task>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            var moveHistory = new List<Move>();
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();

            int moveCount = 0;
            float finalResult = 0;
            string endReason = "进行中";

            var positionHistory = new Dictionary<ulong, int>();
            int noProgressCount = 0;

            while (true)
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        Move? instantKillMove = _generator.GetCaptureKingMove(board);
                        if (instantKillMove != null)
                        {
                            Console.WriteLine($"[规则裁判] 发现对方送将/未应将！执行绝杀: {instantKillMove.Value}");
                            moveHistory.Add(instantKillMove.Value);
                            board.Push(instantKillMove.Value.From, instantKillMove.Value.To);
                            if (onMovePerformed != null)
                            {
                                await onMovePerformed.Invoke(board);
                                await Task.Delay(1000);
                            }
                            endReason = "对方送将/未应将，老将被击杀";
                            finalResult = isRed ? 1.0f : -1.0f;
                            break;
                        }

                        var legalMoves = _generator.GenerateLegalMoves(board);
                        if (legalMoves.Count == 0)
                        {
                            if (onMovePerformed != null)
                                await Task.Delay(1000);
                            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
                            endReason = inCheck ? "绝杀" : "困毙";
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        if (moveCount >= _maxMoves)
                        {
                            if (onMovePerformed != null)
                                await Task.Delay(1000);
                            endReason = "步数限制(强制平局)";
                            finalResult = 0.0f;
                            break;
                        }

                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

                        (Move mctsBestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 3200);

                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        double temperature = (moveCount < _exploreMoves) ? 1.0 : 0.05;
                        Move move = SelectMoveByTemperature(piData, temperature, legalMoves);

                        if (move.From == move.To || move.From < 0 || !legalMoves.Any(m => m.From == move.From && m.To == move.To))
                        {
                            Console.WriteLine($"[警告拦截] 拦截到无效动作 {move.From}->{move.To}，强行重算...");
                            move = legalMoves[_random.Next(legalMoves.Count)];
                        }

                        sbyte pieceToMove = board.GetPiece(move.From);
                        bool isCapture = board.GetPiece(move.To) != 0;
                        bool isPawnAdvance = Math.Abs(pieceToMove) == 7;

                        if (isCapture || isPawnAdvance)
                        {
                            noProgressCount = 0;
                            positionHistory.Clear();
                        }
                        else
                            noProgressCount++;

                        moveHistory.Add(move);
                        board.Push(move.From, move.To);

                        if (onMovePerformed != null)
                            await onMovePerformed.Invoke(board);

                        moveCount++;
                        ulong currentHash = board.CurrentHash;
                        if (!positionHistory.ContainsKey(currentHash))
                            positionHistory[currentHash] = 0;
                        positionHistory[currentHash]++;

                        if (positionHistory[currentHash] >= 3)
                        {
                            endReason = "三次重复局面(平局)";
                            finalResult = 0.0f;
                            break;
                        }
                        if (noProgressCount >= 100)
                        {
                            endReason = "自然限着百步无进展(平局)";
                            finalResult = 0.0f;
                            break;
                        }
                    }
                }
                catch (Exception ex) { Console.WriteLine($"[SelfPlay Error] {ex.Message}"); break; }
            }

            string resultStr = finalResult == 0 ? "平局" : (finalResult > 0 ? "红胜" : "黑胜");
            return new GameResult(FinalizeData(gameHistory, finalResult, board), endReason, resultStr, moveCount, moveHistory);
        }

        private Move SelectMoveByTemperature(float[] piData, double temperature, List<Move> legalMoves)
        {
            var validMoves = new List<(Move move, double prob)>();
            foreach (var m in legalMoves)
            {
                int idx = m.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                    validMoves.Add((m, piData[idx]));
            }

            if (validMoves.Count == 0)
                return legalMoves[_random.Next(legalMoves.Count)];
            if (temperature < 0.1)
                return validMoves.OrderByDescending(x => x.prob).First().move;

            double[] poweredPi = validMoves.Select(x => Math.Pow(x.prob, 1.0 / temperature)).ToArray();
            double sum = poweredPi.Sum();
            if (sum <= 0 || double.IsNaN(sum))
                return validMoves.OrderByDescending(x => x.prob).First().move;

            double r = _random.NextDouble() * sum;
            double cumulative = 0;
            for (int i = 0; i < poweredPi.Length; i++)
            {
                cumulative += poweredPi[i];
                if (r <= cumulative)
                    return validMoves[i].move;
            }
            return validMoves.Last().move;
        }

        private float[] FlipPolicy(float[] originalPi)
        {
            float[] flippedPi = new float[8100];
            for (int i = 0; i < 8100; i++)
            {
                if (originalPi[i] <= 0)
                    continue;
                int from = i / 90, to = i % 90;
                int r1 = from / 9, c1 = from % 9, r2 = to / 9, c2 = to % 9;
                int from_f = (9 - r1) * 9 + (8 - c1);
                int to_f = (9 - r2) * 9 + (8 - c2);
                int idx_f = from_f * 90 + to_f;
                if (idx_f >= 0 && idx_f < 8100)
                    flippedPi[idx_f] = originalPi[i];
            }
            return flippedPi;
        }

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult, Board finalBoard)
        {
            var examples = new List<TrainingExample>(history.Count);
            float adjustedResult = finalResult;

            if (Math.Abs(finalResult) < 0.001f)
            {
                float redMaterial = CalculateMaterialScore(finalBoard, true);
                float blackMaterial = CalculateMaterialScore(finalBoard, false);
                float materialBias = _materialBias;

                if (redMaterial > blackMaterial)
                    adjustedResult = materialBias;
                else if (blackMaterial > redMaterial)
                    adjustedResult = -materialBias;
                else
                    adjustedResult = 0.0f;
            }

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                float valueForCurrentPlayer = step.isRedTurn ? adjustedResult : -adjustedResult;

                // 【核心修复】：将隐式元组改为显式 struct
                var sparsePolicy = step.policy
                                       .Select((p, idx) => new ActionProb(idx, p))
                                       .Where(x => x.Prob > 0)
                                       .ToArray();

                examples.Add(new TrainingExample(step.state, sparsePolicy, valueForCurrentPlayer));
            }
            return examples;
        }

        private float CalculateMaterialScore(Board board, bool isRed)
        {
            float score = 0;
            for (int i = 0; i < 90; i++)
            {
                sbyte p = board.GetPiece(i);
                if (p == 0)
                    continue;
                if ((isRed && p > 0) || (!isRed && p < 0))
                {
                    int type = Math.Abs(p);
                    score += type switch
                    {
                        1 => 0,
                        2 => 2,
                        3 => 2,
                        4 => 4,
                        5 => 9,
                        6 => 4.5f,
                        7 => 1,
                        _ => 0
                    };
                }
            }
            return score;
        }
    }
}