using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System.Diagnostics;

namespace ChineseChessAI.Training
{
    public record GameResult(
        List<TrainingExample> ExamplesA,
        List<TrainingExample> ExamplesB,
        string EndReason,
        string ResultStr,
        int MoveCount,
        List<Move> MoveHistory,
        DateTimeOffset StartedAt,
        DateTimeOffset EndedAt,
        TimeSpan Elapsed,
        bool IsSuccess = true);

    public class SelfPlay
    {
        private readonly MCTSEngine _engineA;
        private readonly MCTSEngine _engineB;
        private readonly ChineseChessRuleEngine _rules;

        private readonly int _maxMoves;
        private readonly int _exploreMoves;
        private readonly float _materialBias;
        private readonly float _earlyDrawPenalty;
        private readonly float _lateDrawPenalty;

        private readonly double _lowTempA;
        private readonly double _lowTempB;
        private readonly int _simsA;
        private readonly int _simsB;
        private readonly RuntimeDiagnostics.RollingCounter _legalMoveMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlayLegalMovesMs", 50);
        private readonly RuntimeDiagnostics.RollingCounter _stateEncodeMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlayStateEncodeMs", 50);
        private readonly RuntimeDiagnostics.RollingCounter _searchMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlaySearchMs", 50);
        private readonly RuntimeDiagnostics.RollingCounter _plyMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlayPlyMs", 50);

        public SelfPlay(
            MCTSEngine engineA,
            MCTSEngine engineB,
            int maxMoves = 150,
            int exploreMoves = 40,
            float materialBias = 0.4f,
            double lowTempA = 0.1,
            double lowTempB = 0.1,
            int simsA = 400,
            int simsB = 400,
            float earlyDrawPenalty = 0.0f,
            float lateDrawPenalty = 0.0f)
        {
            _engineA = engineA;
            _engineB = engineB;
            _rules = new ChineseChessRuleEngine();
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
            DateTimeOffset startedAt = DateTimeOffset.Now;
            var board = new Board();
            board.Reset();

            var moveHistory = new List<Move>();
            var gameHistoryA = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var gameHistoryB = new List<(float[] state, float[] policy, bool isRedTurn)>();

            int moveCount = 0;
            float finalResult = 0;
            string endReason = "进行中";
            bool isSuccess = true;

            var positionHistory = new Dictionary<ulong, int>
            {
                [board.CurrentHash] = 1
            };
            int noProgressCount = 0;

            while (true)
            {
                try
                {
                    var plyStopwatch = Stopwatch.StartNew();
                    if (cancellationToken.IsCancellationRequested)
                    {
                        isSuccess = false;
                        endReason = "训练被强制终止";
                        break;
                    }

                    bool isRed = board.IsRedTurn;
                    bool isEngineA = isRed == engineAIsRed;
                    var activeEngine = isEngineA ? _engineA : _engineB;
                    int activeSims = isEngineA ? _simsA : _simsB;
                    double activeLowTemp = isEngineA ? _lowTempA : _lowTempB;

                    Move? instantKillMove = _rules.GetCaptureKingMove(board);
                    if (instantKillMove != null)
                    {
                        float[] instantStateData;
                        using (var instantStateTensor = StateEncoder.Encode(board))
                        using (var instantState3D = instantStateTensor.squeeze(0))
                        using (var instantStateCpu = instantState3D.cpu())
                        {
                            instantStateData = instantStateCpu.data<float>().ToArray();
                        }

                        float[] instantPiData = new float[8100];
                        instantPiData[instantKillMove.Value.ToNetworkIndex()] = 1.0f;
                        float[] instantTrainingPi = isRed ? instantPiData : StateEncoder.FlipPolicy(instantPiData);

                        if (isEngineA)
                            gameHistoryA.Add((instantStateData, instantTrainingPi, isRed));
                        else
                            gameHistoryB.Add((instantStateData, instantTrainingPi, isRed));

                        Console.WriteLine($"[瑙勫垯瑁佸垽] 鍙戠幇閫佸皢/鏈簲灏嗭紝鎵ц缁濇潃: {instantKillMove.Value}");
                        moveHistory.Add(instantKillMove.Value);
                        board.Push(instantKillMove.Value.From, instantKillMove.Value.To);

                        if (onMovePerformed != null)
                        {
                            await onMovePerformed.Invoke(board);
                            await Task.Delay(1000);
                        }

                        moveCount++;
                        endReason = "瀵规柟閫佸皢/鏈簲灏嗭紝鑰佸皢琚嚮鏉€";
                        finalResult = isRed ? 1.0f : -1.0f;
                        break;
                    }

                    var legalMoveStopwatch = Stopwatch.StartNew();
                    var legalMoves = _rules.GetLegalMoves(board, cancellationToken: cancellationToken);
                    _legalMoveMsCounter.AddSample(legalMoveStopwatch.ElapsedMilliseconds);
                    if (legalMoves.Count == 0)
                    {
                        if (onMovePerformed != null)
                            await Task.Delay(1000);

                        bool inCheck = !_rules.IsKingSafe(board, board.IsRedTurn);
                        endReason = inCheck ? "缁濇潃" : "鍥版瘷";
                        finalResult = board.IsRedTurn ? -1.0f : 1.0f;
                        break;
                    }

                    if (moveCount >= _maxMoves)
                    {
                        if (onMovePerformed != null)
                            await Task.Delay(1000);

                        endReason = "姝ユ暟闄愬埗(寮哄埗骞冲眬)";
                        finalResult = 0.0f;
                        break;
                    }

                    float[] stateData;
                    var stateEncodeStopwatch = Stopwatch.StartNew();
                    using (var stateTensor = StateEncoder.Encode(board))
                    using (var state3D = stateTensor.squeeze(0))
                    using (var stateCpu = state3D.cpu())
                    {
                        stateData = stateCpu.data<float>().ToArray();
                    }
                    _stateEncodeMsCounter.AddSample(stateEncodeStopwatch.ElapsedMilliseconds);

                    var searchStopwatch = Stopwatch.StartNew();
                    (_, float[] piData) = await activeEngine.GetMoveWithProbabilitiesAsArrayAsync(
                        board,
                        activeSims,
                        moveCount,
                        _maxMoves,
                        cancellationToken);
                    _searchMsCounter.AddSample(searchStopwatch.ElapsedMilliseconds);

                    float[] trainingPi = isRed ? piData : StateEncoder.FlipPolicy(piData);
                    if (isEngineA)
                        gameHistoryA.Add((stateData, trainingPi, isRed));
                    else
                        gameHistoryB.Add((stateData, trainingPi, isRed));

                    double temperature = moveCount < _exploreMoves ? 1.0 : activeLowTemp;
                    Move move = SelectMoveByTemperature(piData, temperature, legalMoves);

                    if (move.From == move.To || move.From < 0 || !legalMoves.Any(m => m.From == move.From && m.To == move.To))
                    {
                        Console.WriteLine($"[璀﹀憡鎷︽埅] 鎷︽埅鍒版棤鏁堝姩浣?{move.From}->{move.To}锛屽己琛岄噸閫?..");
                        move = legalMoves[Random.Shared.Next(legalMoves.Count)];
                    }

                    moveHistory.Add(move);
                    board.Push(move.From, move.To);

                    if (onMovePerformed != null)
                        await onMovePerformed.Invoke(board);

                    moveCount++;
                    ulong currentHash = board.CurrentHash;

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

                    _plyMsCounter.AddSample(plyStopwatch.ElapsedMilliseconds);

                    if (positionHistory[currentHash] >= 3)
                    {
                        endReason = "涓夋閲嶅灞€闈?骞冲眬)";
                        finalResult = 0.0f;
                        break;
                    }

                    if (noProgressCount >= 100)
                    {
                        endReason = "鑷劧闄愮潃鐧炬鏃犺繘灞?骞冲眬)";
                        finalResult = 0.0f;
                        break;
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
                    RuntimeDiagnostics.Log($"[SelfPlay异常-堆栈] {ex}");
                    endReason = $"内部错误: {ex.Message}";
                    break;
                }
            }

            string resultStr = isSuccess ? (finalResult == 0 ? "平局" : (finalResult > 0 ? "红胜" : "黑胜")) : "异常中断";
            var examplesA = FinalizeData(gameHistoryA, finalResult, board);
            var examplesB = FinalizeData(gameHistoryB, finalResult, board);
            DateTimeOffset endedAt = DateTimeOffset.Now;
            return new GameResult(examplesA, examplesB, endReason, resultStr, moveCount, moveHistory, startedAt, endedAt, endedAt - startedAt, isSuccess);
        }

        private Move SelectMoveByTemperature(float[] piData, double temperature, List<Move> legalMoves)
        {
            var validMoves = new List<(Move move, double prob)>();
            foreach (var move in legalMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                    validMoves.Add((move, piData[idx]));
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
