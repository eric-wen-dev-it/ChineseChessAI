using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using TorchSharp;

namespace ChineseChessAI.Training
{
    public record TrainingExample(float[] State, float[] Policy, float Value);

    public record GameResult(List<TrainingExample> Examples, string EndReason, string ResultStr, int MoveCount);

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

            // --- 1. 随机开局 ---
            for (int i = 0; i < 4; i++)
            {
                var legalMoves = _generator.GenerateLegalMoves(board);
                if (legalMoves.Count > 0)
                {
                    var randomMove = legalMoves[_random.Next(legalMoves.Count)];
                    board.Push(randomMove.From, randomMove.To);
                    if (onMovePerformed != null)
                        await onMovePerformed.Invoke(board);
                }
            }

            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            int moveCount = board.GetHistory().Count();
            float finalResult = 0;
            string endReason = "进行中";

            // --- 2. 正式对弈循环 ---
            while (true)
            {
                try
                {
                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;

                        // 编码局面：如果是黑方，StateEncoder 内部会自动翻转物理棋盘
                        var stateTensor = StateEncoder.Encode(board);
                        float[] stateData = stateTensor.squeeze(0).cpu().data<float>().ToArray();

                        // MCTS 搜索：返回基于“当前方视角”的概率分布
                        (Move mctsBestMove, float[] piData) = await _engine.GetMoveWithProbabilitiesAsArrayAsync(board, 3200);

                        // 记录样本：Policy 必须对应当前视角，无需额外翻转
                        gameHistory.Add((stateData, piData, isRed));

                        // --- 【关键修复：坐标映射与反转】 ---
                        var legalMoves = _generator.GenerateLegalMoves(board);
                        Move? killMove = legalMoves.FirstOrDefault(m => _generator.CanCaptureKing(board, m));

                        Move move;
                        bool isInstantKill = false;

                        if (killMove != null)
                        {
                            move = killMove.Value;
                            isInstantKill = true;
                        }
                        else
                        {
                            double temperature = (moveCount < 150) ? 1.0 : 0.4;
                            // 从概率分布中采样选出一个基于“视角坐标”的动作
                            move = SelectMoveByTemperature(piData, temperature);

                            // 【核心修复点】如果是黑方，必须将视角坐标映射回逻辑棋盘物理坐标
                            if (!isRed)
                            {
                                move = FlipMove(move);
                            }

                            // 安全拦截：如果采样动作原地踏步或不合法，强制纠正
                            if (move.From == move.To || !legalMoves.Any(m => m.From == move.From && m.To == move.To))
                            {
                                move = legalMoves[_random.Next(legalMoves.Count)];
                            }
                        }

                        // 执行走子
                        board.Push(move.From, move.To);

                        if (onMovePerformed != null)
                            await onMovePerformed.Invoke(board);

                        moveCount++;

                        // --- 3. 终局判定 ---
                        if (isInstantKill)
                        {
                            await Task.Delay(1000); // 终局停留观察
                            endReason = "老将被击杀";
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        var remainingMoves = _generator.GenerateLegalMoves(board);
                        if (remainingMoves.Count == 0)
                        {
                            await Task.Delay(1000);
                            bool inCheck = !_generator.IsKingSafe(board, board.IsRedTurn);
                            endReason = inCheck ? "绝杀" : "困毙";
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                            break;
                        }

                        if (board.GetRepetitionCount() >= 8 || moveCount >= 600)
                        {
                            await Task.Delay(1000);
                            endReason = moveCount >= 600 ? "步数限制" : "八次重复";
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
            return new GameResult(FinalizeData(gameHistory, finalResult, board), endReason, resultStr, moveCount);
        }

        /// <summary>
        /// 执行坐标 180 度反转 (row, col) -> (9-row, 8-col)
        /// </summary>
        private Move FlipMove(Move m)
        {
            int fR = m.From / 9, fC = m.From % 9;
            int tR = m.To / 9, tC = m.To % 9;
            return new Move((9 - fR) * 9 + (8 - fC), (9 - tR) * 9 + (8 - tC));
        }

        private Move SelectMoveByTemperature(float[] piData, double temperature)
        {
            if (temperature < 0.1)
            {
                return Move.FromNetworkIndex(Array.IndexOf(piData, piData.Max()));
            }

            double[] poweredPi = piData.Select(p => Math.Pow(p, 1.0 / temperature)).ToArray();
            double sum = poweredPi.Sum();
            double r = _random.NextDouble() * sum;
            double cumulative = 0;
            for (int i = 0; i < poweredPi.Length; i++)
            {
                cumulative += poweredPi[i];
                if (r <= cumulative)
                    return Move.FromNetworkIndex(i);
            }
            return Move.FromNetworkIndex(Array.IndexOf(piData, piData.Max()));
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