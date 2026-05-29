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
        bool IsSuccess = true,
        float RatingResultForRed = 0.0f);

    public class SelfPlay
    {
        private readonly IGameEngine _engineA;
        private readonly IGameEngine _engineB;
        private readonly ChineseChessRuleEngine _rules;

        private readonly int _maxMoves;
        private readonly int _exploreMoves;
        private readonly float _materialBias;

        private readonly double _lowTempA;
        private readonly double _lowTempB;
        private readonly int _simsA;
        private readonly int _simsB;
        private const int PikafishTeacherNodes = 2000;
        private const int PikafishTeacherStride = 4;
        private const int PikafishTeacherTailPlies = 12;
        private const int PikafishTeacherMaxAnalysesPerSide = 6;
        private static readonly TimeSpan PikafishTeacherAnalysisTimeout = TimeSpan.FromSeconds(3);
        private readonly RuntimeDiagnostics.RollingCounter _legalMoveMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlayLegalMovesMs", 50);
        private readonly RuntimeDiagnostics.RollingCounter _stateEncodeMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlayStateEncodeMs", 50);
        private readonly RuntimeDiagnostics.RollingCounter _searchMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlaySearchMs", 50);
        private readonly RuntimeDiagnostics.RollingCounter _plyMsCounter = new RuntimeDiagnostics.RollingCounter("SelfPlayPlyMs", 50);

        public SelfPlay(
            MCTSEngine engineA,
            MCTSEngine engineB,
            int maxMoves = 150,
            int exploreMoves = 40,
            float materialBias = 0.1f,
            double lowTempA = 0.1,
            double lowTempB = 0.1,
            int simsA = 400,
            int simsB = 400)
        {
            _engineA = new MctsGameEngineAdapter(engineA);
            _engineB = new MctsGameEngineAdapter(engineB);
            _rules = new ChineseChessRuleEngine();
            _maxMoves = maxMoves;
            _exploreMoves = exploreMoves;
            _materialBias = materialBias;
            _lowTempA = lowTempA;
            _lowTempB = lowTempB;
            _simsA = simsA;
            _simsB = simsB;
        }

        public SelfPlay(
            IGameEngine engineA,
            IGameEngine engineB,
            int maxMoves = 150,
            int exploreMoves = 40,
            float materialBias = 0.1f,
            double lowTempA = 0.1,
            double lowTempB = 0.1,
            int simsA = 400,
            int simsB = 400)
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
        }

        public async Task<GameResult> RunGameAsync(bool engineAIsRed, Func<Board, Task>? onMovePerformed = null, System.Threading.CancellationToken cancellationToken = default)
        {
            DateTimeOffset startedAt = DateTimeOffset.Now;
            var board = new Board();
            board.Reset();

            var moveHistory = new List<Move>();
            var moveHistoryUcci = new List<string>();
            var gameHistoryA = new List<PendingTrainingStep>();
            var gameHistoryB = new List<PendingTrainingStep>();

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

                        var ucciBefore = moveHistoryUcci.ToArray();
                        if (isEngineA)
                            gameHistoryA.Add(new PendingTrainingStep(instantStateData, instantTrainingPi, isRed, ucciBefore));
                        else
                            gameHistoryB.Add(new PendingTrainingStep(instantStateData, instantTrainingPi, isRed, ucciBefore));

                        Console.WriteLine($"[瑙勫垯瑁佸垽] 鍙戠幇閫佸皢/鏈簲灏嗭紝鎵ц缁濇潃: {instantKillMove.Value}");
                        moveHistory.Add(instantKillMove.Value);
                        moveHistoryUcci.Add(NotationConverter.MoveToUcci(instantKillMove.Value));
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

                        (finalResult, endReason) = await AdjudicateQuietEndAsync(board, moveHistory, "步数限制", cancellationToken);
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
                    (_, float[] piData) = await activeEngine.GetMoveWithPolicyAsync(
                        board,
                        activeSims,
                        moveCount,
                        _maxMoves,
                        cancellationToken);
                    if (cancellationToken.IsCancellationRequested)
                    {
                        isSuccess = false;
                        endReason = "训练被强制终止";
                        break;
                    }

                    _searchMsCounter.AddSample(searchStopwatch.ElapsedMilliseconds);

                    float[] trainingPi = isRed ? piData : StateEncoder.FlipPolicy(piData);
                    var historyBeforeMove = moveHistoryUcci.ToArray();
                    if (isEngineA)
                        gameHistoryA.Add(new PendingTrainingStep(stateData, trainingPi, isRed, historyBeforeMove));
                    else
                        gameHistoryB.Add(new PendingTrainingStep(stateData, trainingPi, isRed, historyBeforeMove));

                    double temperature = moveCount < _exploreMoves ? 1.0 : activeLowTemp;
                    Move move = SelectMoveByTemperature(board, piData, temperature, legalMoves);

                    if (move.From == move.To || move.From < 0 || !legalMoves.Any(m => m.From == move.From && m.To == move.To))
                    {
                        Console.WriteLine($"[璀﹀憡鎷︽埅] 鎷︽埅鍒版棤鏁堝姩浣?{move.From}->{move.To}锛屽己琛岄噸閫?..");
                        move = legalMoves[Random.Shared.Next(legalMoves.Count)];
                    }

                    moveHistory.Add(move);
                    moveHistoryUcci.Add(NotationConverter.MoveToUcci(move));
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
                        (finalResult, endReason) = await AdjudicateQuietEndAsync(board, moveHistory, "三次重复局面", cancellationToken);
                        break;
                    }

                    if (noProgressCount >= 100)
                    {
                        (finalResult, endReason) = await AdjudicateQuietEndAsync(board, moveHistory, "自然限着百步无进展", cancellationToken);
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
            float ratingResultForRed = isSuccess ? GetRatingResult(finalResult, endReason) : 0.0f;
            if (isSuccess)
            {
                await EnrichWithPikafishTeacherAsync(gameHistoryA, cancellationToken).ConfigureAwait(false);
                await EnrichWithPikafishTeacherAsync(gameHistoryB, cancellationToken).ConfigureAwait(false);
            }

            var examplesA = FinalizeData(gameHistoryA, ratingResultForRed, board);
            var examplesB = FinalizeData(gameHistoryB, ratingResultForRed, board);
            DateTimeOffset endedAt = DateTimeOffset.Now;
            return new GameResult(examplesA, examplesB, endReason, resultStr, moveCount, moveHistory, startedAt, endedAt, endedAt - startedAt, isSuccess, ratingResultForRed);
        }

        private Move SelectMoveByTemperature(Board board, float[] piData, double temperature, List<Move> legalMoves)
        {
            var selectableMoves = legalMoves.Where(move => !board.WillRepeatPosition(move.From, move.To)).ToList();
            if (selectableMoves.Count == 0)
                selectableMoves = legalMoves;

            var validMoves = new List<(Move move, double prob)>();
            foreach (var move in selectableMoves)
            {
                int idx = move.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                    validMoves.Add((move, piData[idx]));
            }

            if (validMoves.Count == 0)
                return selectableMoves[Random.Shared.Next(selectableMoves.Count)];
            if (temperature <= 0.1)
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

        private async Task EnrichWithPikafishTeacherAsync(List<PendingTrainingStep> history, CancellationToken cancellationToken)
        {
            if (history.Count == 0)
                return;

            foreach (int i in SelectPikafishTeacherIndexes(history.Count))
            {
                cancellationToken.ThrowIfCancellationRequested();

                PendingTrainingStep step = history[i];
                PikafishTeacherAnalysis? analysis;
                using (var analysisCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                {
                    analysisCts.CancelAfter(PikafishTeacherAnalysisTimeout);
                    try
                    {
                        analysis = await PikafishAdjudicator.TryAnalyzeAsync(
                            step.UcciHistoryBefore,
                            step.IsRedTurn,
                            PikafishTeacherNodes,
                            analysisCts.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
                    {
                        continue;
                    }
                }

                if (analysis == null)
                    continue;

                step.TeacherValue = analysis.ValueForCurrentPlayer;

                Move? teacherMove = NotationConverter.UcciToMove(analysis.BestMove);
                if (teacherMove.HasValue)
                {
                    float[] teacherPolicy = new float[8100];
                    int idx = teacherMove.Value.ToNetworkIndex();
                    if (idx >= 0 && idx < teacherPolicy.Length)
                    {
                        teacherPolicy[idx] = 1.0f;
                        step.TeacherPolicy = step.IsRedTurn ? teacherPolicy : StateEncoder.FlipPolicy(teacherPolicy);
                    }
                }
            }
        }

        private static List<int> SelectPikafishTeacherIndexes(int historyCount)
        {
            var selected = new SortedSet<int>();
            for (int i = 0; i < historyCount; i += PikafishTeacherStride)
                selected.Add(i);

            int tailStart = Math.Max(0, historyCount - PikafishTeacherTailPlies);
            for (int i = tailStart; i < historyCount; i++)
                selected.Add(i);

            if (selected.Count <= PikafishTeacherMaxAnalysesPerSide)
                return selected.ToList();

            var values = selected.ToList();
            var capped = new SortedSet<int>();
            for (int i = 0; i < PikafishTeacherMaxAnalysesPerSide; i++)
            {
                int sourceIndex = (int)Math.Round(i * (values.Count - 1) / (double)(PikafishTeacherMaxAnalysesPerSide - 1));
                capped.Add(values[sourceIndex]);
            }

            return capped.ToList();
        }

        private List<TrainingExample> FinalizeData(List<PendingTrainingStep> history, float finalResult, Board finalBoard)
        {
            var examples = new List<TrainingExample>(history.Count);
            bool isTrueDraw = Math.Abs(finalResult) < 0.001f;

            for (int i = 0; i < history.Count; i++)
            {
                var step = history[i];
                float valueForCurrentPlayer;
                if (isTrueDraw)
                {
                    valueForCurrentPlayer = 0.0f;
                }
                else
                {
                    valueForCurrentPlayer = step.IsRedTurn ? finalResult : -finalResult;
                }

                var sparsePolicy = step.Policy
                    .Select((p, idx) => new ActionProb(idx, p))
                    .Where(x => x.Prob > 0)
                    .ToArray();

                ActionProb[]? teacherSparsePolicy = step.TeacherPolicy?
                    .Select((p, idx) => new ActionProb(idx, p))
                    .Where(x => x.Prob > 0)
                    .ToArray();

                examples.Add(new TrainingExample(
                    step.State,
                    sparsePolicy,
                    valueForCurrentPlayer,
                    step.TeacherValue,
                    teacherSparsePolicy));
            }

            return examples;
        }

        private sealed class PendingTrainingStep
        {
            public PendingTrainingStep(float[] state, float[] policy, bool isRedTurn, string[] ucciHistoryBefore)
            {
                State = state;
                Policy = policy;
                IsRedTurn = isRedTurn;
                UcciHistoryBefore = ucciHistoryBefore;
            }

            public float[] State { get; }
            public float[] Policy { get; }
            public bool IsRedTurn { get; }
            public string[] UcciHistoryBefore { get; }
            public float? TeacherValue { get; set; }
            public float[]? TeacherPolicy { get; set; }
        }

        private static string ComposeAdjudicationReason(string baseReason, float adjudicatedResult)
        {
            if (adjudicatedResult > 0.5f)
                return $"{baseReason}(子力裁判红胜)";
            if (adjudicatedResult < -0.5f)
                return $"{baseReason}(子力裁判黑胜)";
            return $"{baseReason}(平局)";
        }

        private static async Task<(float Result, string Reason)> AdjudicateQuietEndAsync(
            Board board,
            List<Move> moveHistory,
            string baseReason,
            CancellationToken cancellationToken)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            float materialResult = BoardEvaluation.AdjudicateDrawByMaterial(board);
            return (materialResult, ComposeAdjudicationReason(baseReason, materialResult));
        }

        private float GetRatingResult(float finalResult, string endReason)
        {
            if (Math.Abs(finalResult) < 0.001f)
                return 0.0f;

            if (!IsWeakAdjudication(endReason))
                return finalResult;

            // An adjudicated win is useful, but it should not train or rate
            // like a completed kill. This pushes agents toward finishing won games.
            float adjudicatedWinValue = Math.Clamp(_materialBias, 0.05f, 0.5f);
            return MathF.CopySign(adjudicatedWinValue, finalResult);
        }

        private static bool IsWeakAdjudication(string endReason)
        {
            return endReason.Contains("子力裁判", StringComparison.Ordinal)
                || endReason.Contains("Pikafish评估裁判", StringComparison.Ordinal);
        }
    }
}
