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
    public record TrainingExample(float[] State, float[] Policy, float Value);

    public record GameResult(List<TrainingExample> Examples, string EndReason, string ResultStr, int MoveCount, List<Move> MoveHistory);

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

        public async Task<GameResult> RunGameAsync(Func<Board, Task>? onMovePerformed = null)
        {
            var board = new Board();
            board.Reset();

            var moveHistory = new List<Move>();

            // --- 1. 随机开局 (增加开局多样性) ---
            for (int i = 0; i < 4; i++)
            {
                var initMoves = _generator.GenerateLegalMoves(board);
                if (initMoves.Count > 0)
                {
                    var randomMove = initMoves[_random.Next(initMoves.Count)];

                    moveHistory.Add(randomMove); // 记录随机开局走法
                    board.Push(randomMove.From, randomMove.To);

                    if (onMovePerformed != null)
                        await onMovePerformed.Invoke(board);
                }
            }

            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = board.GetHistory().Count();
            float finalResult = 0;
            string endReason = "进行中";

            // === 【裁判的记忆本】 ===
            var positionHistory = new Dictionary<ulong, int>();
            int noProgressCount = 0;

            // --- 2. 正式对弈循环 ---
            while (true)
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        // --- 【防线 1：送将必败拦截】 ---
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

                        // --- 获取合法走法 ---
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

                        // --- 兜底防线：防止宇宙级死循环 ---
                        if (moveCount >= 600)
                        {
                            if (onMovePerformed != null)
                                await Task.Delay(1000);

                            endReason = "步数限制(强制平局)";
                            finalResult = 0.0f;
                            break;
                        }

                        // --- MCTS 神经网络思考 ---
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

                        (Move mctsBestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 3200);

                        float[] trainingPi = isRed ? piData : FlipPolicy(piData);
                        gameHistory.Add((stateData, trainingPi, isRed));

                        double temperature = (moveCount < 150) ? 1.0 : 0.4;
                        Move move = SelectMoveByTemperature(piData, temperature, legalMoves);

                        if (move.From == move.To || move.From < 0 || !legalMoves.Any(m => m.From == move.From && m.To == move.To))
                        {
                            Console.WriteLine($"[警告拦截] 拦截到无效动作 {move.From}->{move.To}，强行重算...");
                            move = legalMoves[_random.Next(legalMoves.Count)];
                        }

                        // === 【铁血裁判机制 1：不可逆动作检测】 ===
                        sbyte pieceToMove = board.GetPiece(move.From);
                        bool isCapture = board.GetPiece(move.To) != 0;
                        bool isPawnAdvance = Math.Abs(pieceToMove) == 7;

                        if (isCapture || isPawnAdvance)
                        {
                            noProgressCount = 0;
                            positionHistory.Clear();
                        }
                        else
                        {
                            noProgressCount++;
                        }

                        // 执行走法
                        moveHistory.Add(move);
                        board.Push(move.From, move.To);

                        if (onMovePerformed != null)
                            await onMovePerformed.Invoke(board);

                        moveCount++;

                        // === 【铁血裁判机制 2：三次重复局面判平】 ===
                        // 修正：调用 Board.cs 中的 CurrentHash 而不是 ZobristKey
                        ulong currentHash = board.CurrentHash;
                        if (!positionHistory.ContainsKey(currentHash))
                        {
                            positionHistory[currentHash] = 0;
                        }
                        positionHistory[currentHash]++;

                        if (positionHistory[currentHash] >= 3)
                        {
                            endReason = "三次重复局面(平局)";
                            finalResult = 0.0f;
                            break;
                        }

                        // === 【铁血裁判机制 3：自然限着规则】 ===
                        if (noProgressCount >= 100)
                        {
                            endReason = "自然限着百步无进展(平局)";
                            finalResult = 0.0f;
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[SelfPlay Error] {ex.Message}");
                    break;
                }
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
                {
                    validMoves.Add((m, piData[idx]));
                }
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
            var examples = new List<TrainingExample>();
            float adjustedResult = finalResult;

            if (Math.Abs(finalResult) < 0.001f)
            {
                float redMaterial = CalculateMaterialScore(finalBoard, true);
                float blackMaterial = CalculateMaterialScore(finalBoard, false);
                if (redMaterial > blackMaterial)
                    adjustedResult = 0.15f;
                else if (blackMaterial > redMaterial)
                    adjustedResult = -0.15f;
                else
                    adjustedResult = -0.1f;
            }

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                float valueForCurrentPlayer = step.isRedTurn ? adjustedResult : -adjustedResult;
                examples.Add(new TrainingExample(step.state, step.policy, valueForCurrentPlayer));
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