using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Traditional;
using ChineseChessAI.Training;

string command = args.Length > 0 ? args[0].ToLowerInvariant() : "help";
string repoRoot = FindRepoRoot();

switch (command)
{
    case "book":
        BuildBook(repoRoot, args);
        break;
    case "bench":
        await RunDepthBench(args);
        break;
    case "bench-classic":
        await RunClassicBench(args);
        break;
    case "bench-mcts":
        await RunMctsBench(args);
        break;
    case "search":
        RunSearch(args);
        break;
    default:
        PrintHelp();
        break;
}

static void BuildBook(string repoRoot, string[] args)
{
    int maxPly = GetIntArg(args, "--ply", 24);
    int orderPly = GetIntArg(args, "--order-ply", 80);
    int orderMinCount = GetIntArg(args, "--order-min-count", 2);
    int orderTopMoves = GetIntArg(args, "--order-top", 4);
    int maxGames = GetIntArg(args, "--games", int.MaxValue);
    string source = GetStringArg(args, "--source", FindDefaultBookSource(repoRoot));
    string output = GetStringArg(args, "--out", Path.Combine(repoRoot, "data", "opening_book.json"));
    string orderOutput = GetStringArg(args, "--order-out", Path.Combine(repoRoot, "data", "master_move_ordering.json"));

    var book = new OpeningBook(maxPly);
    int games = book.LoadFromPath(source, maxGames);
    book.SaveCache(output);

    var board = new Board();
    bool hit = book.TryGetMove(board, OpeningBookMode.Best, out var firstMove);
    Console.WriteLine($"source={source}");
    Console.WriteLine($"games={games}");
    Console.WriteLine($"positions={book.PositionCount}");
    Console.WriteLine($"output={output}");
    Console.WriteLine($"initial_hit={hit}");
    if (hit)
        Console.WriteLine($"initial_best={firstMove}");

    foreach (var entry in book.GetBookMoves(board, 8))
        Console.WriteLine($"initial_move={entry.Move} count={entry.Count}");

    if (!string.Equals(orderOutput, "off", StringComparison.OrdinalIgnoreCase))
    {
        var orderingBook = new OpeningBook(orderPly);
        int orderingGames = orderingBook.LoadFromPath(source, maxGames);
        orderingBook.Prune(orderMinCount, orderTopMoves);
        orderingBook.SaveCache(orderOutput);
        Console.WriteLine($"ordering_games={orderingGames}");
        Console.WriteLine($"ordering_positions={orderingBook.PositionCount}");
        Console.WriteLine($"ordering_output={orderOutput}");
        Console.WriteLine($"ordering_min_count={orderMinCount}");
        Console.WriteLine($"ordering_top_moves={orderTopMoves}");
    }
}

static async Task RunDepthBench(string[] args)
{
    int games = GetIntArg(args, "--games", 2);
    int maxMoves = GetIntArg(args, "--moves", 80);
    int lowDepth = GetIntArg(args, "--low", 3);
    int highDepth = GetIntArg(args, "--high", 4);
    var score = new MatchScore();

    for (int i = 0; i < games; i++)
    {
        bool highIsRed = i % 2 == 0;
        var options = CreateCurrentTraditionalOptions();
        var high = new TraditionalGameEngineAdapter(new TraditionalEngine(options));
        var low = new TraditionalGameEngineAdapter(new TraditionalEngine(options));
        var engineA = highIsRed ? high : low;
        var engineB = highIsRed ? low : high;
        int budgetA = highIsRed ? highDepth : lowDepth;
        int budgetB = highIsRed ? lowDepth : highDepth;

        var result = await RunMatch(engineA, engineB, budgetA, budgetB, maxMoves, TimeSpan.FromMinutes(10));
        var outcome = ClassifyResult(result.ResultStr, highIsRed);
        score.Add(outcome.AWon, outcome.BWon);
        Console.WriteLine($"game={i + 1} high_red={highIsRed} result={result.ResultStr} moves={result.MoveCount} reason={result.EndReason}");
    }

    Console.WriteLine($"summary high_depth={highDepth} low_depth={lowDepth} high_wins={score.AWins} low_wins={score.BWins} draws={score.Draws}");
}

static async Task RunClassicBench(string[] args)
{
    int games = GetIntArg(args, "--games", 2);
    int maxMoves = GetIntArg(args, "--moves", 80);
    int depth = GetIntArg(args, "--depth", 4);
    var score = new MatchScore();

    for (int i = 0; i < games; i++)
    {
        bool currentIsRed = i % 2 == 0;
        var current = new TraditionalGameEngineAdapter(new TraditionalEngine(CreateCurrentTraditionalOptions()));
        var classic = new TraditionalGameEngineAdapter(new TraditionalEngine(CreateClassicTraditionalOptions()));
        var engineA = currentIsRed ? current : classic;
        var engineB = currentIsRed ? classic : current;

        var result = await RunMatch(engineA, engineB, depth, depth, maxMoves, TimeSpan.FromMinutes(10));
        var outcome = ClassifyResult(result.ResultStr, currentIsRed);
        score.Add(outcome.AWon, outcome.BWon);
        Console.WriteLine($"game={i + 1} current_red={currentIsRed} result={result.ResultStr} moves={result.MoveCount} reason={result.EndReason}");
    }

    Console.WriteLine($"summary depth={depth} current_wins={score.AWins} classic_wins={score.BWins} draws={score.Draws}");
}

static async Task RunMctsBench(string[] args)
{
    string modelPath = GetStringArg(args, "--model", string.Empty);
    if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
    {
        Console.WriteLine("bench-mcts requires --model PATH.");
        return;
    }

    int games = GetIntArg(args, "--games", 2);
    int maxMoves = GetIntArg(args, "--moves", 80);
    int depth = GetIntArg(args, "--depth", 4);
    int simulations = GetIntArg(args, "--sims", 64);
    var score = new MatchScore();

    using var model = new CChessNet();
    ModelManager.LoadModel(model, modelPath);

    for (int i = 0; i < games; i++)
    {
        bool traditionalIsRed = i % 2 == 0;
        using var mcts = new MCTSEngine(model, batchSize: 16);
        var traditional = new TraditionalGameEngineAdapter(new TraditionalEngine(CreateCurrentTraditionalOptions()));
        var mctsAdapter = new MctsGameEngineAdapter(mcts);
        var engineA = traditionalIsRed ? (IGameEngine)traditional : mctsAdapter;
        var engineB = traditionalIsRed ? mctsAdapter : (IGameEngine)traditional;
        int budgetA = traditionalIsRed ? depth : simulations;
        int budgetB = traditionalIsRed ? simulations : depth;

        var result = await RunMatch(engineA, engineB, budgetA, budgetB, maxMoves, TimeSpan.FromMinutes(15));
        var outcome = ClassifyResult(result.ResultStr, traditionalIsRed);
        score.Add(outcome.AWon, outcome.BWon);
        Console.WriteLine($"game={i + 1} traditional_red={traditionalIsRed} result={result.ResultStr} moves={result.MoveCount} reason={result.EndReason}");
    }

    Console.WriteLine($"summary traditional_depth={depth} mcts_sims={simulations} traditional_wins={score.AWins} mcts_wins={score.BWins} draws={score.Draws}");
}

static void RunSearch(string[] args)
{
    int depth = GetIntArg(args, "--depth", 5);
    int moveTimeMs = GetIntArg(args, "--time", 5000);
    string movesText = GetStringArg(args, "--moves", string.Empty);

    var board = new Board();
    var generator = new MoveGenerator();
    foreach (string ucci in movesText.Split(' ', StringSplitOptions.RemoveEmptyEntries))
    {
        Move? parsed = ChineseChessAI.Utils.NotationConverter.UcciToMove(ucci);
        if (!parsed.HasValue)
            throw new InvalidOperationException($"Invalid UCCI move: {ucci}");

        var legalMoves = generator.GenerateLegalMoves(board, skipPerpetualCheck: false);
        if (!legalMoves.Contains(parsed.Value))
            throw new InvalidOperationException($"Illegal move at current position: {ucci}");

        board.Push(parsed.Value.From, parsed.Value.To);
    }

    var engine = new TraditionalEngine(CreateCurrentTraditionalOptions());
    var result = engine.Search(board, new SearchLimits(depth, moveTimeMs, 4));
    Console.WriteLine($"side={(board.IsRedTurn ? "red" : "black")}");
    Console.WriteLine($"depth={result.Depth}");
    Console.WriteLine($"score={result.Score}");
    Console.WriteLine($"nodes={result.Nodes}");
    Console.WriteLine($"time_ms={result.Elapsed.TotalMilliseconds:F0}");
    Console.WriteLine($"completed={result.Completed}");
    Console.WriteLine($"bestmove={result.BestMove}");
}

static async Task<GameResult> RunMatch(
    IGameEngine redEngine,
    IGameEngine blackEngine,
    int redBudget,
    int blackBudget,
    int maxMoves,
    TimeSpan timeout)
{
    var selfPlay = new SelfPlay(redEngine, blackEngine, maxMoves, 0, 0, 0.1, 0.1, redBudget, blackBudget);
    using var cts = new CancellationTokenSource(timeout);
    return await selfPlay.RunGameAsync(engineAIsRed: true, cancellationToken: cts.Token);
}

static TraditionalEngineOptions CreateCurrentTraditionalOptions()
{
    var book = OpeningBook.LoadDefaultCache(maxPly: 24);
    return new TraditionalEngineOptions
    {
        OpeningBook = book,
        OpeningBookMode = book.PositionCount > 0 ? OpeningBookMode.Weighted : OpeningBookMode.Off,
        MoveOrderingBook = OpeningBook.LoadDefaultCache(maxPly: 80, fileName: "master_move_ordering.json")
    };
}

static TraditionalEngineOptions CreateClassicTraditionalOptions()
{
    return new TraditionalEngineOptions
    {
        OpeningBookMode = OpeningBookMode.Off,
        UseNullMovePruning = false,
        UseFutilityPruning = false,
        UseRazoring = false,
        UseSeePruning = false,
        MateSearchPly = 1
    };
}

static (bool AWon, bool BWon) ClassifyResult(string result, bool aIsRed)
{
    bool redWon = result == "红胜";
    bool blackWon = result == "黑胜";
    return ((aIsRed && redWon) || (!aIsRed && blackWon), (aIsRed && blackWon) || (!aIsRed && redWon));
}

static int GetIntArg(string[] args, string name, int fallback)
{
    int index = Array.IndexOf(args, name);
    if (index >= 0 && index + 1 < args.Length && int.TryParse(args[index + 1], out int value))
        return value;
    return fallback;
}

static string GetStringArg(string[] args, string name, string fallback)
{
    int index = Array.IndexOf(args, name);
    return index >= 0 && index + 1 < args.Length ? args[index + 1] : fallback;
}

static string FindRepoRoot()
{
    string dir = AppDomain.CurrentDomain.BaseDirectory;
    for (int i = 0; i < 8; i++)
    {
        if (File.Exists(Path.Combine(dir, "ChineseChessAI.csproj")))
            return dir;
        dir = Path.GetFullPath(Path.Combine(dir, ".."));
    }

    return Directory.GetCurrentDirectory();
}

static string FindDefaultBookSource(string repoRoot)
{
    string pgn = Path.Combine(repoRoot, "xqdb_masters_40711_UCI_games.pgn");
    if (File.Exists(pgn))
        return pgn;

    return Path.Combine(repoRoot, "data", "master_data");
}

static void PrintHelp()
{
    Console.WriteLine("TraditionalTools book [--source DIR|PGN] [--out FILE] [--ply 24] [--games N]");
    Console.WriteLine("                      [--order-out FILE|off] [--order-ply 80] [--order-min-count 2] [--order-top 4]");
    Console.WriteLine("TraditionalTools bench [--games 2] [--moves 80] [--low 3] [--high 4]");
    Console.WriteLine("TraditionalTools bench-classic [--games 2] [--moves 80] [--depth 4]");
    Console.WriteLine("TraditionalTools bench-mcts --model PATH [--games 2] [--moves 80] [--depth 4] [--sims 64]");
    Console.WriteLine("TraditionalTools search --moves \"h2e2 h7e7\" [--depth 5] [--time 5000]");
}

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
