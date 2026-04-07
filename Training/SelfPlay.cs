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
    public record GameResult(List<TrainingExample> ExamplesA, List<TrainingExample> ExamplesB, string EndReason, string ResultStr, int MoveCount, List<Move> MoveHistory, bool IsSuccess = true);

    public class SelfPlay
    {
        private readonly MCTSEngine _engineA;
        private readonly MCTSEngine _engineB;
        private readonly MoveGenerator _generator;

        private readonly int _maxMoves;
        private readonly int _exploreMoves;
        private readonly float _materialBias;
        private readonly float _earlyDrawPenalty;
        private readonly float _lateDrawPenalty;
        
        // --- 个性化参数 ---
        private readonly double _lowTempA;
        private readonly double _lowTempB;
        private readonly int _simsA;
        private readonly int _simsB;

        public SelfPlay(MCTSEngine engineA, MCTSEngine engineB, int maxMoves = 150, int exploreMoves = 40, float materialBias = 0.4f,
                        double lowTempA = 0.1, double lowTempB = 0.1, int simsA = 400, int simsB = 400,
                        float earlyDrawPenalty = 0.0f, float lateDrawPenalty = 0.0f)
        {
            _engineA = engineA;
            _engineB = engineB;
            _generator = new MoveGenerator();
            _maxMoves = maxMoves;
            _exploreMoves = exploreMoves;
            _materialBias = materialBias;
            _lowTempA = lowTempA;
            _lowTempB = lowTempB;
            _simsA = simsA;
            _simsB = simsB;
            _earlyDrawPenalty = earlyDrawPenalty;
            _lateDrawPenalty = lateDrawPenalty;
        }

        public async Task<GameResult> RunGameAsync(bool engineAIsRed, Func<Board, Task>? onMovePerformed = null, System.Threading.CancellationToken cancellationToken = default)
        {
            var board = new Board();
            board.Reset();

            var moveHistory = new List<Move>();
            var gameHistoryA = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var gameHistoryB = new List<(float[] state, float[] policy, bool isRedTurn)>();

            int moveCount = 0;
            float finalResult = 0;
            string endReason = "进行中";
            bool isSuccess = true;

            var positionHistory = new Dictionary<ulong, int>();
            positionHistory[board.CurrentHash] = 1; // 【修复 P1】：初始局面入记录
            int noProgressCount = 0;

            while (true)
            {
                try
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        isSuccess = false;
                        endReason = "训练被强制终止";
                        break;
                    }

                    using (var moveScope = torch.NewDisposeScope())
                    {
                        bool isRed = board.IsRedTurn;
                        bool isEngineA = (isRed == engineAIsRed);
                        var activeEngine = isEngineA ? _engineA : _engineB;
                        int activeSims = isEngineA ? _simsA : _simsB;
                        double activeLowTemp = isEngineA ? _lowTempA : _lowTempB;

                        Move? instantKillMove = _generator.GetCaptureKingMove(board);
                        if (instantKillMove != null)
                        {
                            // 【修复 P2 #5】：执行绝杀前，必须先记录当前状态，否则模型学不到绝杀动作
                            var instantStateTensor = StateEncoder.Encode(board);
                            float[] instantStateData = instantStateTensor.squeeze(0).cpu().data<float>().ToArray();
                            
                            float[] instantPiData = new float[8100];
                            instantPiData[instantKillMove.Value.ToNetworkIndex()] = 1.0f; // 绝杀步概率设为 1.0
                            float[] instantTrainingPi = isRed ? instantPiData : StateEncoder.FlipPolicy(instantPiData);

                            if (isEngineA)
                                gameHistoryA.Add((instantStateData, instantTrainingPi, isRed));
                            else
                                gameHistoryB.Add((instantStateData, instantTrainingPi, isRed));

                            Console.WriteLine($"[规则裁判] 发现对方送将/未应将！执行绝杀: {instantKillMove.Value}");
                            moveHistory.Add(instantKillMove.Value);
                            board.Push(instantKillMove.Value.From, instantKillMove.Value.To);
                            if (onMovePerformed != null)
                            {
                                await onMovePerformed.Invoke(board);
                                await Task.Delay(1000);
                            }
                            moveCount++;
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
                            finalResult = board.IsRedTurn ? -1.0f : 1.0f; // 纠正：无路可走即输 (BUG 1)
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

                        // 【核心修复 BUG-5】：使用基因传入的 activeSims，而非硬编码的 800
                        (_, float[] piData) = await activeEngine.GetMoveWithProbabilitiesAsArrayAsync(board, activeSims, moveCount, _maxMoves, cancellationToken);

                        float[] trainingPi = isRed ? piData : StateEncoder.FlipPolicy(piData);
                        
                        if (isEngineA)
                            gameHistoryA.Add((stateData, trainingPi, isRed));
                        else
                            gameHistoryB.Add((stateData, trainingPi, isRed));

                        // 使用该智能体特有的探索温度
                        double temperature = (moveCount < _exploreMoves) ? 1.0 : activeLowTemp;
                        Move move = SelectMoveByTemperature(piData, temperature, legalMoves);

                        if (move.From == move.To || move.From < 0 || !legalMoves.Any(m => m.From == move.From && m.To == move.To))
                        {
                            Console.WriteLine($"[警告拦截] 拦截到无效动作 {move.From}->{move.To}，强行重算...");
                            move = legalMoves[Random.Shared.Next(legalMoves.Count)];
                        }

                        moveHistory.Add(move);
                        board.Push(move.From, move.To); // 此处 board.Push 会自动处理不可逆招法导致的哈希清理

                        if (onMovePerformed != null)
                            await onMovePerformed.Invoke(board);

                        moveCount++;
                        ulong currentHash = board.CurrentHash;
                        
                        // 【BUG 1 & 2 修复】：基于 Board 状态精准驱动计数重置
                        if (board.LastMoveWasIrreversible)
                        {
                            noProgressCount = 0;
                            positionHistory.Clear();
                        }
                        else
                        {
                            noProgressCount++;
                        }

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
                catch (OperationCanceledException)
                {
                    isSuccess = false;
                    endReason = "训练被强制终止";
                    break;
                }
                catch (Exception ex)
                {
                    isSuccess = false;
                    endReason = $"内部错误: {ex.Message}";
                    break;
                }
            }

            string resultStr = isSuccess ? (finalResult == 0 ? "平局" : (finalResult > 0 ? "红胜" : "黑胜")) : "异常中断";
            var examplesA = FinalizeData(gameHistoryA, finalResult, board);
            var examplesB = FinalizeData(gameHistoryB, finalResult, board);
            return new GameResult(examplesA, examplesB, endReason, resultStr, moveCount, moveHistory, isSuccess);
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
                return legalMoves[Random.Shared.Next(legalMoves.Count)];
            if (temperature < 0.1)
                return validMoves.OrderByDescending(x => x.prob).First().move;

            double[] poweredPi = validMoves.Select(x => Math.Pow(x.prob, 1.0 / temperature)).ToArray();
            double sum = poweredPi.Sum();
            if (sum <= 0 || double.IsNaN(sum))
                return validMoves.OrderByDescending(x => x.prob).First().move;

            double r = Random.Shared.NextDouble() * sum;
            double cumulative = 0;
            for (int i = 0; i < poweredPi.Length; i++)
            {
                cumulative += poweredPi[i];
                if (r <= cumulative)
                    return validMoves[i].move;
            }
            return validMoves.Last().move;
        }

        private List<TrainingExample> FinalizeData(List<(float[] state, float[] policy, bool isRedTurn)> history, float finalResult, Board finalBoard)
        {
            var examples = new List<TrainingExample>(history.Count);
            float adjustedResult = finalResult;
            bool isSymmetricDrawPenalty = false;

            if (Math.Abs(finalResult) < 0.001f)
            {
                float redMaterial = TrainingOrchestrator.CalculateMaterialScore(finalBoard, true);
                float blackMaterial = TrainingOrchestrator.CalculateMaterialScore(finalBoard, false);

                if (redMaterial > blackMaterial)
                    adjustedResult = _materialBias;
                else if (blackMaterial > redMaterial)
                    adjustedResult = -_materialBias;
                else
                    isSymmetricDrawPenalty = true; 
            }

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                float valueForCurrentPlayer;
                if (isSymmetricDrawPenalty)
                {
                    float progress = (float)i / Math.Max(1, history.Count - 1);
                    valueForCurrentPlayer = progress > 0.5f ? _lateDrawPenalty : _earlyDrawPenalty;
                }
                else
                {
                    valueForCurrentPlayer = step.isRedTurn ? adjustedResult : -adjustedResult;
                }

                var sparsePolicy = step.policy
                                       .Select((p, idx) => new ActionProb(idx, p))
                                       .Where(x => x.Prob > 0)
                                       .ToArray();

                examples.Add(new TrainingExample(step.state, sparsePolicy, valueForCurrentPlayer));
            }
            return examples;
        }
    }
}
