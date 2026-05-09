using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Traditional;
using ChineseChessAI.Utils;
using TorchSharp;

if (args.Length >= 4 && string.Equals(args[0], "match", StringComparison.OrdinalIgnoreCase))
{
    return await RunAgentMatchCommand(args);
}

if (args.Length >= 3 && args[0].EndsWith(".pt", StringComparison.OrdinalIgnoreCase) && args[1].EndsWith(".pt", StringComparison.OrdinalIgnoreCase))
{
    return await RunAgentMatchCommand(["match", .. args]);
}

const string DefaultModelPath = @"D:\Temp\agent_26\agent_26.pt";

string modelPath = args.Length > 0 ? args[0] : DefaultModelPath;
if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Model file not found: {modelPath}");
    return 2;
}

Console.WriteLine($"Model: {modelPath}");
Console.WriteLine($"CUDA available: {torch.cuda.is_available()}");

try
{
    RunMctsVsTraditionalSmoke(modelPath);
    Console.WriteLine("PASS");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine("FAIL");
    Console.Error.WriteLine(ex);
    return 1;
}

static void RunMctsVsTraditionalSmoke(string modelPath)
{
    using var model = new CChessNet(autoCuda: false);
    ModelManager.LoadModel(model, modelPath);
    model.to(torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU);
    model.eval();

    using var mcts = new MCTSEngine(model, batchSize: 4, cPuct: 1.6);
    var traditional = new TraditionalEngine(new TraditionalEngineOptions
    {
        RootParallelism = 1,
        SkipPerpetualCheckInsideSearch = true,
        MateSearchPly = 1
    });
    var rules = new ChineseChessRuleEngine();
    var board = new Board();
    board.Reset();

    AssertMctsMoveIsLegal(mcts, rules, board, moveNumber: 0, maxMoves: 120);
    AssertMctsMoveIsLegal(mcts, rules, board, moveNumber: 120, maxMoves: 120);

    for (int ply = 0; ply < 24; ply++)
    {
        var legalMoves = rules.GetLegalMoves(board);
        if (legalMoves.Count == 0)
        {
            Console.WriteLine($"Game ended before ply {ply}: no legal moves.");
            return;
        }

        Move move;
        string side = board.IsRedTurn ? "red" : "black";
        if (ply % 2 == 0)
        {
            move = GetMctsMove(mcts, rules, board, ply, maxMoves: 120);
            Console.WriteLine($"{ply + 1,2}. MCTS {side}: {move}");
        }
        else
        {
            var result = traditional.Search(board, SearchLimits.FixedDepth(2));
            move = result.BestMove;
            Console.WriteLine($"{ply + 1,2}. Traditional {side}: {move} depth={result.Depth} completed={result.Completed}");
        }

        if (!legalMoves.Any(m => m.From == move.From && m.To == move.To))
        {
            throw new InvalidOperationException(
                $"Illegal move at ply {ply}: {move}. legalMoves={legalMoves.Count}, side={side}, hash={board.CurrentHash}");
        }

        board.Push(move.From, move.To);
    }
}

static void AssertMctsMoveIsLegal(MCTSEngine mcts, ChineseChessRuleEngine rules, Board board, int moveNumber, int maxMoves)
{
    _ = GetMctsMove(mcts, rules, board, moveNumber, maxMoves);
}

static Move GetMctsMove(MCTSEngine mcts, ChineseChessRuleEngine rules, Board board, int moveNumber, int maxMoves)
{
    var legalMoves = rules.GetLegalMoves(board);
    Console.WriteLine($"MCTS probe: moveNumber={moveNumber}, maxMoves={maxMoves}, legalMoves={legalMoves.Count}, redTurn={board.IsRedTurn}, hash={board.CurrentHash}");
    if (legalMoves.Count == 0)
        throw new InvalidOperationException("MCTS probe has no legal moves.");

    try
    {
        var (move, policy) = mcts.GetMoveWithProbabilitiesAsArrayAsync(
                board,
                simulations: 16,
                currentMoves: moveNumber,
                maxMoves: maxMoves,
                CancellationToken.None,
                addRootNoise: false)
            .GetAwaiter()
            .GetResult();

        int nonZeroPolicy = policy.Count(p => p > 0);
        Console.WriteLine($"MCTS result: {move}, nonZeroPolicy={nonZeroPolicy}");
        return move;
    }
    catch
    {
        Console.Error.WriteLine($"MCTS failed: moveNumber={moveNumber}, maxMoves={maxMoves}, legalMoves={legalMoves.Count}, redTurn={board.IsRedTurn}, hash={board.CurrentHash}");
        throw;
    }
}

static async Task<int> RunAgentMatchCommand(string[] args)
{
    string modelAPath = ResolveModelPath(args[1]);
    string modelBPath = ResolveModelPath(args[2]);
    string outputDir = args[3];
    int games = GetIntArg(args, "--games", 100);
    int maxMoves = GetIntArg(args, "--moves", 150);
    int simulations = GetIntArg(args, "--sims", 64);
    int timeoutMinutes = GetIntArg(args, "--timeout-min", 30);
    int batchSize = GetIntArg(args, "--batch", 16);

    Directory.CreateDirectory(outputDir);
    string runStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
    string logPath = Path.Combine(outputDir, $"agent_match_{runStamp}.log");

    void Log(string message)
    {
        string line = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} {message}";
        Console.WriteLine(line);
        File.AppendAllText(logPath, line + Environment.NewLine);
    }

    try
    {
        if (!File.Exists(modelAPath))
            throw new FileNotFoundException("Model A not found.", modelAPath);
        if (!File.Exists(modelBPath))
            throw new FileNotFoundException("Model B not found.", modelBPath);

        Log($"model_a={modelAPath}");
        Log($"model_b={modelBPath}");
        Log($"output={outputDir}");
        Log($"games={games} moves={maxMoves} sims={simulations} batch={batchSize} timeout_min={timeoutMinutes}");
        Log($"cuda_available={torch.cuda.is_available()}");

        using var modelA = new CChessNet(autoCuda: false);
        using var modelB = new CChessNet(autoCuda: false);
        ModelManager.LoadModel(modelA, modelAPath);
        ModelManager.LoadModel(modelB, modelBPath);

        var device = torch.cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;
        modelA.to(device);
        modelB.to(device);
        modelA.eval();
        modelB.eval();

        var score = new MatchScore();
        var rules = new ChineseChessRuleEngine();

        for (int game = 1; game <= games; game++)
        {
            bool agentAIsRed = game % 2 == 1;
            using var engineA = new MCTSEngine(modelA, batchSize: batchSize, cPuct: 2.5);
            using var engineB = new MCTSEngine(modelB, batchSize: batchSize, cPuct: 2.5);
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(timeoutMinutes));

            Log($"game_start game={game} agent_a_red={agentAIsRed}");
            AgentGameResult result = await RunAgentGame(
                game,
                engineA,
                engineB,
                rules,
                agentAIsRed,
                simulations,
                maxMoves,
                timeout.Token,
                Log);

            var outcome = ClassifyAgentMatchResult(result.Result, agentAIsRed);
            score.Add(outcome.AWon, outcome.BWon);
            Log($"game_end game={game} result={result.Result} moves={result.Moves} reason={result.Reason} agent_a_wins={score.AWins} agent_b_wins={score.BWins} draws={score.Draws}");
        }

        Log($"summary games={games} agent_a_wins={score.AWins} agent_b_wins={score.BWins} draws={score.Draws} log={logPath}");
        return 0;
    }
    catch (Exception ex)
    {
        string crashPath = Path.Combine(outputDir, $"agent_match_crash_{runStamp}.log");
        File.WriteAllText(crashPath, ex.ToString());
        Log($"fatal crash_log={crashPath}");
        Log(ex.ToString());
        return 1;
    }
}

static async Task<AgentGameResult> RunAgentGame(
    int game,
    MCTSEngine engineA,
    MCTSEngine engineB,
    ChineseChessRuleEngine rules,
    bool agentAIsRed,
    int simulations,
    int maxMoves,
    CancellationToken cancellationToken,
    Action<string> log)
{
    var board = new Board();
    board.Reset();
    var positionHistory = new Dictionary<ulong, int> { [board.CurrentHash] = 1 };
    int noProgressCount = 0;

    for (int ply = 0; ; ply++)
    {
        cancellationToken.ThrowIfCancellationRequested();

        bool isRed = board.IsRedTurn;
        bool useAgentA = isRed == agentAIsRed;
        string side = isRed ? "red" : "black";
        string agent = useAgentA ? "agent_a" : "agent_b";
        var activeEngine = useAgentA ? engineA : engineB;

        var captureKingMove = rules.GetCaptureKingMove(board);
        if (captureKingMove.HasValue)
        {
            board.Push(captureKingMove.Value.From, captureKingMove.Value.To);
            string result = isRed ? "red_win" : "black_win";
            string ucci = NotationConverter.MoveToUcci(captureKingMove.Value);
            log($"move game={game} ply={ply + 1} agent={agent} side={side} move={ucci} instant_capture_king=true hash={board.CurrentHash}");
            return new AgentGameResult(result, ply + 1, "capture_king");
        }

        var legalMoves = rules.GetLegalMoves(board, cancellationToken: cancellationToken);
        if (legalMoves.Count == 0)
        {
            bool inCheck = !rules.IsKingSafe(board, board.IsRedTurn);
            string result = board.IsRedTurn ? "black_win" : "red_win";
            return new AgentGameResult(result, ply, inCheck ? "checkmate" : "stalemate");
        }

        if (ply >= maxMoves)
            return AdjudicateAgentMatchByMaterial(board, "max_moves", ply);

        log($"search_start game={game} ply={ply + 1} agent={agent} side={side} legal={legalMoves.Count} hash={board.CurrentHash}");
        var started = DateTimeOffset.Now;
        var (move, policy) = await activeEngine.GetMoveWithProbabilitiesAsArrayAsync(
            board,
            simulations,
            ply,
            maxMoves,
            cancellationToken,
            addRootNoise: false);
        TimeSpan elapsed = DateTimeOffset.Now - started;

        if (!legalMoves.Any(m => m.From == move.From && m.To == move.To))
        {
            int policyNonZero = policy.Count(p => p > 0);
            throw new InvalidOperationException(
                $"Illegal MCTS move. game={game} ply={ply + 1} agent={agent} side={side} move={move} legal={legalMoves.Count} policy_nonzero={policyNonZero} hash={board.CurrentHash}");
        }

        board.Push(move.From, move.To);
        string moveUcci = NotationConverter.MoveToUcci(move);
        log($"move game={game} ply={ply + 1} agent={agent} side={side} move={moveUcci} search_ms={elapsed.TotalMilliseconds:F0} hash={board.CurrentHash}");

        if (board.LastMoveWasIrreversible)
        {
            noProgressCount = 0;
            positionHistory.Clear();
        }
        else
        {
            noProgressCount++;
        }

        if (!positionHistory.ContainsKey(board.CurrentHash))
            positionHistory[board.CurrentHash] = 0;
        positionHistory[board.CurrentHash]++;

        if (positionHistory[board.CurrentHash] >= 3)
            return AdjudicateAgentMatchByMaterial(board, "threefold", ply + 1);

        if (noProgressCount >= 100)
            return AdjudicateAgentMatchByMaterial(board, "no_progress_100", ply + 1);
    }
}

static AgentGameResult AdjudicateAgentMatchByMaterial(Board board, string baseReason, int plies)
{
    float adjudication = BoardEvaluation.AdjudicateDrawByMaterial(board);
    if (adjudication > 0.5f)
        return new AgentGameResult("red_win", plies, $"{baseReason}_material_red");
    if (adjudication < -0.5f)
        return new AgentGameResult("black_win", plies, $"{baseReason}_material_black");
    return new AgentGameResult("draw", plies, baseReason);
}

static (bool AWon, bool BWon) ClassifyAgentMatchResult(string result, bool agentAIsRed)
{
    bool redWon = result == "red_win";
    bool blackWon = result == "black_win";
    return ((agentAIsRed && redWon) || (!agentAIsRed && blackWon), (agentAIsRed && blackWon) || (!agentAIsRed && redWon));
}

static int GetIntArg(string[] args, string name, int fallback)
{
    int index = Array.IndexOf(args, name);
    if (index >= 0 && index + 1 < args.Length && int.TryParse(args[index + 1], out int value))
        return value;
    return fallback;
}

static string ResolveModelPath(string path)
{
    if (File.Exists(path))
        return Path.GetFullPath(path);

    if (Path.IsPathRooted(path))
        return path;

    string[] candidateRoots =
    [
        Directory.GetCurrentDirectory(),
        AppDomain.CurrentDomain.BaseDirectory,
        Path.Combine(Directory.GetCurrentDirectory(), "bin", "Debug", "net10.0-windows", "data", "models", "league"),
        Path.Combine(Directory.GetCurrentDirectory(), "bin", "Release", "net10.0-windows", "data", "models", "league")
    ];

    foreach (string root in candidateRoots)
    {
        string candidate = Path.Combine(root, path);
        if (File.Exists(candidate))
            return Path.GetFullPath(candidate);
    }

    return path;
}

readonly record struct AgentGameResult(string Result, int Moves, string Reason);

sealed class MatchScore
{
    public int AWins { get; private set; }
    public int BWins { get; private set; }
    public int Draws { get; private set; }

    public void Add(bool aWon, bool bWon)
    {
        if (aWon)
            AWins++;
        else if (bWon)
            BWins++;
        else
            Draws++;
    }
}
