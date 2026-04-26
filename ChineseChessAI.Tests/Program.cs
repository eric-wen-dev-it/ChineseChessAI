using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Traditional;
using ChineseChessAI.Utils;
using TorchSharp;

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
