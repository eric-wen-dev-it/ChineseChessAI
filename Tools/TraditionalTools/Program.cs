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
    case "knowledge":
        BuildKnowledgeBook(repoRoot, args);
        break;
    case "bench":
        await RunDepthBench(args);
        break;
    case "bench-classic":
        await RunClassicBench(args);
        break;
    case "bench-qs":
        await RunQuiescenceBench(args);
        break;
    case "bench-mcts":
        await RunMctsBench(args);
        break;
    case "search":
        RunSearch(args);
        break;
    case "bench-tt":
        RunTtSelfTest(args);
        break;
    case "engine":
        RunEngineRepl(args);
        break;
    case "engine-mcts":
        await RunMctsRepl(args);
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

static void BuildKnowledgeBook(string repoRoot, string[] args)
{
    int maxPly = GetIntArg(args, "--ply", 120);
    int minCount = GetIntArg(args, "--min-count", 2);
    int topMoves = GetIntArg(args, "--top", 6);
    int maxGames = GetIntArg(args, "--games", int.MaxValue);
    string source = GetStringArg(args, "--source", FindDefaultBookSource(repoRoot));
    string output = GetStringArg(args, "--out", Path.Combine(repoRoot, "data", "master_knowledge_book.json"));

    var knowledge = new MasterKnowledgeBook(maxPly);
    int games = knowledge.LoadFromPath(source, maxGames);
    knowledge.Prune(minCount, topMoves);
    knowledge.SaveCache(output);

    var board = new Board();
    Console.WriteLine($"source={source}");
    Console.WriteLine($"games={games}");
    Console.WriteLine($"result_red_wins={knowledge.RedWinGames}");
    Console.WriteLine($"result_black_wins={knowledge.BlackWinGames}");
    Console.WriteLine($"result_draws={knowledge.DrawGames}");
    Console.WriteLine($"result_unknown={knowledge.UnknownGames}");
    Console.WriteLine($"positions={knowledge.PositionCount}");
    Console.WriteLine($"output={output}");
    Console.WriteLine($"min_count={minCount}");
    Console.WriteLine($"top_moves={topMoves}");
    foreach (var entry in knowledge.GetMoves(board, 8))
    {
        Console.WriteLine($"initial_move={entry.Move} count={entry.Count} red_wins={entry.RedWins} black_wins={entry.BlackWins} draws={entry.Draws} unknown={entry.Unknown}");
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
    bool enhanced = Array.IndexOf(args, "--enhanced") >= 0;
    var score = new MatchScore();

    string dev = TorchSharp.torch.cuda.is_available() ? "CUDA" : "CPU";
    Console.WriteLine($"device={dev} model={System.IO.Path.GetFileName(modelPath)} " +
        $"trad_depth={depth} enhanced={enhanced} sims={simulations}");
    using var model = new CChessNet();
    ModelManager.LoadModel(model, modelPath);

    for (int i = 0; i < games; i++)
    {
        bool traditionalIsRed = i % 2 == 0;
        using var mcts = new MCTSEngine(model, batchSize: 16);
        var tradOpts = CreateCurrentTraditionalOptions();
        if (enhanced) tradOpts = tradOpts.WithEnhancedQuiescence();
        var traditional = new TraditionalGameEngineAdapter(new TraditionalEngine(tradOpts));
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

// 增强档 vs 现行档,同深度互弈。增强档=现行档 + 静搜增强全开(qsearch TT/
// delta/SEE 剪枝/深层纯吃子延伸),用于验证 FC 对照处方的净胜率方向。
static async Task RunQuiescenceBench(string[] args)
{
    int games = GetIntArg(args, "--games", 20);
    int maxMoves = GetIntArg(args, "--moves", 120);
    int depth = GetIntArg(args, "--depth", 4);
    // 消融开关:定位增强档胜率异常时逐项关闭。
    bool noTT = Array.IndexOf(args, "--no-tt") >= 0;
    bool noDelta = Array.IndexOf(args, "--no-delta") >= 0;
    bool noSee = Array.IndexOf(args, "--no-see") >= 0;
    int checkPlies = GetIntArg(args, "--checkplies", 2);
    var score = new MatchScore();

    TraditionalEngineOptions BuildEnhanced()
    {
        var full = CreateCurrentTraditionalOptions();
        return new TraditionalEngineOptions
        {
            OpeningBook = full.OpeningBook,
            OpeningBookMode = full.OpeningBookMode,
            MoveOrderingBook = full.MoveOrderingBook,
            MasterKnowledgeBook = full.MasterKnowledgeBook,
            UseQuiescenceTT = !noTT,
            UseQuiescenceDeltaPruning = !noDelta,
            UseQuiescenceSeePruning = !noSee,
            QuiescenceCheckPlies = checkPlies
        };
    }

    Console.WriteLine($"enhanced_config tt={!noTT} delta={!noDelta} see={!noSee} checkplies={checkPlies}");
    for (int i = 0; i < games; i++)
    {
        bool enhancedIsRed = i % 2 == 0;
        var enhanced = new TraditionalGameEngineAdapter(new TraditionalEngine(BuildEnhanced()));
        var current = new TraditionalGameEngineAdapter(new TraditionalEngine(CreateCurrentTraditionalOptions()));
        var engineA = enhancedIsRed ? enhanced : current;
        var engineB = enhancedIsRed ? current : enhanced;

        var result = await RunMatch(engineA, engineB, depth, depth, maxMoves, TimeSpan.FromMinutes(10));
        var outcome = ClassifyResult(result.ResultStr, enhancedIsRed);
        score.Add(outcome.AWon, outcome.BWon);
        Console.WriteLine($"game={i + 1} enhanced_red={enhancedIsRed} result={result.ResultStr} moves={result.MoveCount} reason={result.EndReason}");
    }

    Console.WriteLine($"summary depth={depth} enhanced_wins={score.AWins} current_wins={score.BWins} draws={score.Draws}");
}

// 无锁置换表自检:①打包往返(边界值+吃杀分 ply 校正逐位);②并发撕裂读安全
// (单写者固定载荷 H,多写者向同一槽位灌不同哈希制造 data churn,读 H 命中必须
// 逐位等于固定载荷,否则=撕裂读逃过 XOR 校验)。用于给"毒表"回归上锁。
static void RunTtSelfTest(string[] args)
{
    int failures = 0;
    const int mate = 1_000_000;

    // ---- ① 打包往返 ----
    var table = new TranspositionTable(1 << 20);
    var cases = new (ulong hash, int depth, int score, Move move, TTBound bound, int ply)[]
    {
        (1001, 0,   0,          new Move(0, 0),   TTBound.Exact, 0),
        (1002, 511, 1_999_999,  new Move(89, 89), TTBound.Lower, 0),
        (1003, 20,  -1_999_999, new Move(89, 0),  TTBound.Upper, 0),
        (1004, 7,   250,        new Move(3, 44),  TTBound.Exact, 0),
        (1005, 12,  -37,        new Move(45, 3),  TTBound.Lower, 0),
        (1006, 9,   mate - 3,   new Move(10, 20), TTBound.Exact, 5),   // 吃杀分,ply 校正
        (1007, 9,   -(mate - 3),new Move(20, 10), TTBound.Upper, 5),
    };
    foreach (var c in cases)
        table.Store(c.hash, c.depth, c.score, c.move, c.bound, c.ply, mate);
    foreach (var c in cases)
    {
        if (!table.TryGet(c.hash, c.ply, mate, out var e))
        {
            Console.WriteLine($"FAIL roundtrip miss hash={c.hash}");
            failures++;
            continue;
        }
        if (e.Depth != c.depth || e.Score != c.score || e.Bound != c.bound
            || e.BestMove.From != c.move.From || e.BestMove.To != c.move.To)
        {
            Console.WriteLine($"FAIL roundtrip mismatch hash={c.hash} got depth={e.Depth} score={e.Score} bound={e.Bound} move={e.BestMove.From}->{e.BestMove.To}");
            failures++;
        }
    }
    Console.WriteLine($"roundtrip cases={cases.Length} failures_so_far={failures}");

    // ---- ② 并发撕裂读安全 ----
    int size = 1 << 14;
    var ct = new TranspositionTable(size);
    ulong H = 12345;                       // 固定载荷键
    var payload = (depth: 5, score: 100, move: new Move(10, 20), bound: TTBound.Exact);
    long tornReads = 0;
    long hHits = 0;
    int seconds = GetIntArg(args, "--seconds", 3);
    var stop = System.Diagnostics.Stopwatch.StartNew();
    long deadlineMs = seconds * 1000L;

    var threads = new List<Thread>();
    // 固定单写者:反复用同一载荷写 H。
    threads.Add(new Thread(() =>
    {
        while (stop.ElapsedMilliseconds < deadlineMs)
            ct.Store(H, payload.depth, payload.score, payload.move, payload.bound, 0, mate);
    }));
    // 污染写者:向 H 所在槽位灌不同哈希(H + k*size 同索引),制造 data churn。
    for (int t = 0; t < 4; t++)
    {
        int seed = t;
        threads.Add(new Thread(() =>
        {
            ulong h = H + (ulong)(size * (seed + 1));
            var rnd = new Random(seed * 7919 + 1);
            while (stop.ElapsedMilliseconds < deadlineMs)
            {
                ct.Store(h, rnd.Next(1, 40), rnd.Next(-500_000, 500_000),
                    new Move(rnd.Next(0, 90), rnd.Next(0, 90)), (TTBound)rnd.Next(0, 3), 0, mate);
                h += (ulong)size; // 保持同一索引
            }
        }));
    }
    // 读者:读 H,命中必须逐位等于固定载荷。
    for (int t = 0; t < 4; t++)
    {
        threads.Add(new Thread(() =>
        {
            while (stop.ElapsedMilliseconds < deadlineMs)
            {
                if (ct.TryGet(H, 0, mate, out var e))
                {
                    Interlocked.Increment(ref hHits);
                    if (e.Depth != payload.depth || e.Score != payload.score
                        || e.Bound != payload.bound
                        || e.BestMove.From != payload.move.From || e.BestMove.To != payload.move.To)
                    {
                        Interlocked.Increment(ref tornReads);
                    }
                }
            }
        }));
    }

    foreach (var th in threads) th.Start();
    foreach (var th in threads) th.Join();

    Console.WriteLine($"concurrency seconds={seconds} h_hits={hHits} torn_reads={tornReads}");
    if (tornReads > 0)
    {
        Console.WriteLine("FAIL torn reads detected");
        failures++;
    }

    Console.WriteLine(failures == 0 ? "PASS bench-tt" : $"FAIL bench-tt failures={failures}");
}

static void RunSearch(string[] args)
{
    int depth = GetIntArg(args, "--depth", 5);
    int moveTimeMs = GetIntArg(args, "--time", 5000);
    int quiescenceDepth = GetIntArg(args, "--qdepth", 4);
    bool enhanced = Array.IndexOf(args, "--enhanced") >= 0;
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

    var baseOptions = CreateCurrentTraditionalOptions();
    var engine = new TraditionalEngine(enhanced ? baseOptions.WithEnhancedQuiescence() : baseOptions);
    var result = engine.Search(board, new SearchLimits(depth, moveTimeMs, quiescenceDepth));
    Console.WriteLine($"side={(board.IsRedTurn ? "red" : "black")}");
    Console.WriteLine($"depth={result.Depth}");
    Console.WriteLine($"score={result.Score}");
    Console.WriteLine($"nodes={result.Nodes}");
    Console.WriteLine($"time_ms={result.Elapsed.TotalMilliseconds:F0}");
    Console.WriteLine($"completed={result.Completed}");
    Console.WriteLine($"bestmove={result.BestMove}");
}

// 常驻引擎 REPL(FC 桥接用):stdin 每行 "go <ucci...>"(自初始局面的完整着法
// 序列),stdout 回一行 "bestmove <ucci> score S depth D nodes N";无合法着法回
// "bestmove none ..."。谱只加载一次,避免每手重启进程的秒级开销。
static void RunEngineRepl(string[] args)
{
    int depth = GetIntArg(args, "--depth", 6);
    int moveTimeMs = GetIntArg(args, "--time", 3000);
    int quiescenceDepth = GetIntArg(args, "--qdepth", 12);
    bool enhanced = Array.IndexOf(args, "--enhanced") >= 0;

    var stdout = new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true };
    Console.SetOut(stdout);

    var baseOptions = CreateCurrentTraditionalOptions();
    var engine = new TraditionalEngine(enhanced ? baseOptions.WithEnhancedQuiescence() : baseOptions);
    var generator = new MoveGenerator();
    Console.WriteLine($"ready depth={depth} time={moveTimeMs} qdepth={quiescenceDepth} enhanced={enhanced}");

    string? line;
    while ((line = Console.ReadLine()) != null)
    {
        line = line.Trim();
        if (line.Length == 0)
            continue;
        if (line == "quit")
            break;
        if (!line.StartsWith("go", StringComparison.Ordinal))
        {
            Console.WriteLine("error unknown-command");
            continue;
        }

        try
        {
            var board = new Board();
            bool bad = false;
            string badMove = string.Empty;
            foreach (string ucci in line.Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1))
            {
                Move? parsed = ChineseChessAI.Utils.NotationConverter.UcciToMove(ucci);
                if (!parsed.HasValue || !generator.GenerateLegalMoves(board, skipPerpetualCheck: false).Contains(parsed.Value))
                {
                    bad = true;
                    badMove = ucci;
                    break;
                }

                board.Push(parsed.Value.From, parsed.Value.To);
            }

            if (bad)
            {
                Console.WriteLine($"error illegal {badMove}");
                continue;
            }

            if (generator.GenerateLegalMoves(board, skipPerpetualCheck: false).Count == 0)
            {
                Console.WriteLine("bestmove none score -1000000 depth 0 nodes 0");
                continue;
            }

            var result = engine.Search(board, new SearchLimits(depth, moveTimeMs, quiescenceDepth));
            Console.WriteLine($"bestmove {result.BestMove} score {result.Score} depth {result.Depth} nodes {result.Nodes}");
        }
        catch (Exception e)
        {
            Console.WriteLine("error " + e.Message.Replace('\r', ' ').Replace('\n', ' '));
        }
    }
}

// 常驻神经引擎 REPL(FC 桥接用,与 engine 同协议):stdin "go <ucci...>",
// stdout "bestmove <ucci> score S depth D nodes N"(神经无 α-β 分,S/D/N 置 0)。
// 加载一次模型,addRootNoise=false 对弈用确定性策略。
static async Task RunMctsRepl(string[] args)
{
    string modelPath = GetStringArg(args, "--model", string.Empty);
    if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
    {
        Console.WriteLine("error engine-mcts requires --model PATH");
        return;
    }
    int sims = GetIntArg(args, "--sims", 800);
    int batch = GetIntArg(args, "--batch", 16);

    var stdout = new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true };
    Console.SetOut(stdout);

    using var model = new CChessNet();
    ModelManager.LoadModel(model, modelPath);
    using var mcts = new MCTSEngine(model, batchSize: batch);
    var generator = new MoveGenerator();
    string dev = TorchSharp.torch.cuda.is_available() ? "CUDA" : "CPU";
    Console.WriteLine($"ready mcts sims={sims} device={dev} model={System.IO.Path.GetFileName(modelPath)}");

    string? line;
    while ((line = Console.ReadLine()) != null)
    {
        line = line.Trim();
        if (line.Length == 0) continue;
        if (line == "quit") break;
        if (!line.StartsWith("go", StringComparison.Ordinal))
        {
            Console.WriteLine("error unknown-command");
            continue;
        }

        try
        {
            var board = new Board();
            bool bad = false; string badMove = string.Empty;
            foreach (string ucci in line.Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1))
            {
                Move? parsed = ChineseChessAI.Utils.NotationConverter.UcciToMove(ucci);
                if (!parsed.HasValue || !generator.GenerateLegalMoves(board, skipPerpetualCheck: false).Contains(parsed.Value))
                {
                    bad = true; badMove = ucci; break;
                }
                board.Push(parsed.Value.From, parsed.Value.To);
            }
            if (bad) { Console.WriteLine($"error illegal {badMove}"); continue; }
            if (generator.GenerateLegalMoves(board, skipPerpetualCheck: false).Count == 0)
            {
                Console.WriteLine("bestmove none score -1000000 depth 0 nodes 0");
                continue;
            }

            var (move, _) = await mcts.GetMoveWithProbabilitiesAsArrayAsync(
                board, sims, currentMoves: 0, maxMoves: 999,
                cancellationToken: default, addRootNoise: false);
            Console.WriteLine($"bestmove {move} score 0 depth 0 nodes {sims}");
        }
        catch (Exception e)
        {
            Console.WriteLine("error " + e.Message.Replace('\r', ' ').Replace('\n', ' '));
        }
    }
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
        MoveOrderingBook = OpeningBook.LoadDefaultCache(maxPly: 80, fileName: "master_move_ordering.json"),
        MasterKnowledgeBook = MasterKnowledgeBook.LoadDefaultCache(maxPly: 120)
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
    Console.WriteLine("TraditionalTools bench-qs [--games 20] [--moves 120] [--depth 4]");
    Console.WriteLine("TraditionalTools search --moves \"h2e2 h7e7\" [--depth 5] [--time 5000] [--qdepth 4] [--enhanced]");
    Console.WriteLine("TraditionalTools bench-tt [--seconds 3]");
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
