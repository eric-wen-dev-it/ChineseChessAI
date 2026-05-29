using ChineseChessAI.Utils;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;

namespace ChineseChessAI.Training
{
    public sealed record PikafishAdjudication(float ResultForRed, string ScoreText, int? Depth, string BestMove, bool IsMate);

    public sealed record PikafishTeacherAnalysis(
        float ValueForCurrentPlayer,
        float ValueForRed,
        string BestMove,
        string ScoreText,
        int? Depth,
        bool IsMate);

    public static class PikafishAdjudicator
    {
        private const int DefaultNodes = 5000;
        private const int WinThresholdCentipawns = 250;
        private const int MaxClientCount = 4;
        private static readonly SemaphoreSlim ClientSlots = new(MaxClientCount, MaxClientCount);
        private static readonly ConcurrentBag<PikafishEvaluationClient> Clients = new();
        private static readonly TimeSpan InitializationRetryDelay = TimeSpan.FromSeconds(30);
        private static readonly object InitializationStateLock = new();
        private static bool _initializationFailed;
        private static DateTimeOffset _nextInitializationAttemptUtc = DateTimeOffset.MinValue;

        public static async Task<PikafishAdjudication?> TryAdjudicateAsync(
            IReadOnlyList<string> ucciHistory,
            bool redToMove,
            CancellationToken cancellationToken)
        {
            PikafishTeacherAnalysis? analysis = await TryAnalyzeAsync(
                ucciHistory,
                redToMove,
                DefaultNodes,
                cancellationToken).ConfigureAwait(false);

            if (analysis == null)
                return null;

            if (analysis.IsMate)
            {
                float result = analysis.ValueForRed > 0 ? 1.0f : -1.0f;
                return new PikafishAdjudication(
                    result,
                    analysis.ScoreText,
                    analysis.Depth,
                    analysis.BestMove,
                    IsMate: true);
            }

            float adjudicated = analysis.ValueForRed >= CentipawnsToValue(WinThresholdCentipawns)
                ? 1.0f
                : analysis.ValueForRed <= CentipawnsToValue(-WinThresholdCentipawns)
                    ? -1.0f
                    : 0.0f;

            return new PikafishAdjudication(
                adjudicated,
                analysis.ScoreText,
                analysis.Depth,
                analysis.BestMove,
                IsMate: false);
        }

        public static async Task<PikafishTeacherAnalysis?> TryAnalyzeAsync(
            IReadOnlyList<string> ucciHistory,
            bool redToMove,
            int nodes,
            CancellationToken cancellationToken)
        {
            if (!CanAttemptInitialization())
                return null;

            await ClientSlots.WaitAsync(cancellationToken).ConfigureAwait(false);
            PikafishEvaluationClient? client = null;
            bool returnClientToPool = false;
            try
            {
                client = await RentClientAsync(cancellationToken).ConfigureAwait(false);
                if (client == null)
                    return null;

                PikafishSearchScore? score = await client.EvaluateAsync(
                    ucciHistory,
                    redToMove,
                    nodes,
                    cancellationToken).ConfigureAwait(false);
                returnClientToPool = true;

                if (score == null)
                    return null;

                if (score.BestMove == "0000" || score.BestMove == "(none)")
                {
                    const float valueForCurrentPlayer = -1.0f;
                    float valueForRed = redToMove ? valueForCurrentPlayer : -valueForCurrentPlayer;
                    return new PikafishTeacherAnalysis(
                        valueForCurrentPlayer,
                        valueForRed,
                        score.BestMove,
                        "bestmove none",
                        score.Depth,
                        IsMate: true);
                }

                if (score.MatePly.HasValue)
                {
                    bool rootSideWins = score.MatePly.Value > 0;
                    float valueForCurrentPlayer = rootSideWins ? 1.0f : -1.0f;
                    float valueForRed = redToMove ? valueForCurrentPlayer : -valueForCurrentPlayer;
                    return new PikafishTeacherAnalysis(
                        valueForCurrentPlayer,
                        valueForRed,
                        score.BestMove,
                        $"mate {score.MatePly.Value}",
                        score.Depth,
                        IsMate: true);
                }

                if (!score.Centipawns.HasValue)
                    return null;

                float valueForCurrent = CentipawnsToValue(score.Centipawns.Value);
                float valueForRedFromCp = redToMove ? valueForCurrent : -valueForCurrent;
                int redCentipawns = redToMove ? score.Centipawns.Value : -score.Centipawns.Value;
                return new PikafishTeacherAnalysis(
                    valueForCurrent,
                    valueForRedFromCp,
                    score.BestMove,
                    $"cp {redCentipawns}",
                    score.Depth,
                    IsMate: false);
            }
            catch (OperationCanceledException)
            {
                client?.Dispose();
                returnClientToPool = false;
                throw;
            }
            catch (Exception ex)
            {
                RuntimeDiagnostics.Log($"[Pikafish裁判] 不可用，回退子力裁判: {ex.Message}");
                client?.Dispose();
                returnClientToPool = false;
                return null;
            }
            finally
            {
                if (returnClientToPool && client != null)
                    Clients.Add(client);
                ClientSlots.Release();
            }
        }

        private static async Task<PikafishEvaluationClient?> RentClientAsync(CancellationToken cancellationToken)
        {
            if (Clients.TryTake(out var client))
                return client;

            return await CreateClientAsync(cancellationToken).ConfigureAwait(false);
        }

        private static bool CanAttemptInitialization()
        {
            lock (InitializationStateLock)
            {
                return !_initializationFailed || DateTimeOffset.UtcNow >= _nextInitializationAttemptUtc;
            }
        }

        private static void MarkInitializationSucceeded()
        {
            lock (InitializationStateLock)
            {
                _initializationFailed = false;
                _nextInitializationAttemptUtc = DateTimeOffset.MinValue;
            }
        }

        private static void MarkInitializationFailed()
        {
            lock (InitializationStateLock)
            {
                _initializationFailed = true;
                _nextInitializationAttemptUtc = DateTimeOffset.UtcNow + InitializationRetryDelay;
            }
        }

        private static float CentipawnsToValue(int centipawns)
        {
            return Math.Clamp(MathF.Tanh(centipawns / 600.0f), -0.99f, 0.99f);
        }

        private static async Task<PikafishEvaluationClient?> CreateClientAsync(CancellationToken cancellationToken)
        {
            List<string> enginePaths = ResolveEnginePaths().ToList();
            if (enginePaths.Count == 0)
            {
                RuntimeDiagnostics.Log("[Pikafish裁判] 未找到 Pikafish 可执行文件，回退子力裁判。");
                MarkInitializationFailed();
                return null;
            }

            Exception? lastError = null;
            foreach (string enginePath in enginePaths)
            {
                PikafishEvaluationClient? client = null;
                try
                {
                    client = new PikafishEvaluationClient(enginePath);
                    await client.InitializeAsync(cancellationToken).ConfigureAwait(false);
                    RuntimeDiagnostics.Log($"[Pikafish裁判] 已加载: {enginePath}");
                    MarkInitializationSucceeded();
                    return client;
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    client?.Dispose();
                    lastError = ex;
                    RuntimeDiagnostics.Log($"[Pikafish裁判] 启动失败，尝试下一个: {enginePath} | {ex.Message}");
                }
            }

            MarkInitializationFailed();
            if (lastError != null)
                RuntimeDiagnostics.Log($"[Pikafish裁判] 全部候选启动失败，回退子力裁判: {lastError.Message}");
            return null;
        }

        private static IEnumerable<string> ResolveEnginePaths()
        {
            string? configured = Environment.GetEnvironmentVariable("PIKAFISH_PATH");
            if (!string.IsNullOrWhiteSpace(configured) && File.Exists(configured))
                yield return Path.GetFullPath(configured);

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string runtimePikafishDir = Path.Combine(baseDir, "Pikafish", "Windows");
            string[] candidates =
            {
                Path.Combine(runtimePikafishDir, "pikafish-avx2.exe"),
                Path.Combine(runtimePikafishDir, "pikafish-avxvnni.exe"),
                Path.Combine(runtimePikafishDir, "pikafish-bmi2.exe"),
                Path.Combine(runtimePikafishDir, "pikafish-sse41-popcnt.exe"),
                Path.Combine(runtimePikafishDir, "pikafish-avx512.exe"),
                Path.Combine(runtimePikafishDir, "pikafish-avx512icl.exe"),
                Path.Combine(runtimePikafishDir, "pikafish-vnni512.exe")
            };

            foreach (string candidate in candidates.Where(File.Exists).Distinct(StringComparer.OrdinalIgnoreCase))
                yield return candidate;
        }
    }

    internal sealed record PikafishSearchScore(int? Centipawns, int? MatePly, int? Depth, string BestMove);

    internal sealed class PikafishEvaluationClient : IDisposable
    {
        private static readonly Regex ScoreRegex = new(@"score\s+(cp|mate)\s+(-?\d+)", RegexOptions.Compiled | RegexOptions.IgnoreCase);
        private static readonly Regex DepthRegex = new(@"depth\s+(\d+)", RegexOptions.Compiled | RegexOptions.IgnoreCase);
        private readonly Process _process;
        private readonly ConcurrentQueue<string> _lines = new();
        private readonly SemaphoreSlim _lineSignal = new(0);
        private readonly CancellationTokenSource _readerCts = new();
        private readonly Task _stdoutTask;
        private readonly Task _stderrTask;

        public PikafishEvaluationClient(string enginePath)
        {
            _process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = enginePath,
                    WorkingDirectory = ResolveWorkingDirectory(enginePath),
                    UseShellExecute = false,
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                },
                EnableRaisingEvents = true
            };

            if (!_process.Start())
                throw new InvalidOperationException("Failed to start Pikafish process.");

            _stdoutTask = Task.Run(() => ReadLoopAsync(_process.StandardOutput, _readerCts.Token));
            _stderrTask = Task.Run(() => ReadLoopAsync(_process.StandardError, _readerCts.Token));
        }

        public async Task InitializeAsync(CancellationToken cancellationToken)
        {
            SendLine("uci");
            await WaitForLineAsync(line => line == "uciok" || line == "ucciok", TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
            SendLine("setoption name Threads value 1");
            SendLine("setoption name Hash value 64");
            await WaitReadyAsync(cancellationToken).ConfigureAwait(false);
        }

        public async Task<PikafishSearchScore?> EvaluateAsync(
            IReadOnlyList<string> ucciHistory,
            bool redToMove,
            int nodes,
            CancellationToken cancellationToken)
        {
            ClearPendingLines();
            SendLine("ucinewgame");
            await WaitReadyAsync(cancellationToken).ConfigureAwait(false);

            string moves = ucciHistory.Count == 0 ? string.Empty : " moves " + string.Join(' ', ucciHistory);
            SendLine("position startpos" + moves);
            SendLine($"go nodes {Math.Max(1, nodes)}");

            int? centipawns = null;
            int? matePly = null;
            int? depth = null;
            string bestMove = string.Empty;

            using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);

            while (true)
            {
                string line;
                try
                {
                    line = await WaitForAnyLineAsync(linkedCts.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
                {
                    SendLine("stop");
                    throw new TimeoutException("Timed out waiting for Pikafish adjudication.");
                }

                if (line.StartsWith("info ", StringComparison.OrdinalIgnoreCase))
                {
                    var scoreMatch = ScoreRegex.Match(line);
                    if (scoreMatch.Success && int.TryParse(scoreMatch.Groups[2].Value, out int scoreValue))
                    {
                        if (scoreMatch.Groups[1].Value.Equals("cp", StringComparison.OrdinalIgnoreCase))
                        {
                            centipawns = scoreValue;
                            matePly = null;
                        }
                        else
                        {
                            matePly = scoreValue;
                            centipawns = null;
                        }
                    }

                    var depthMatch = DepthRegex.Match(line);
                    if (depthMatch.Success && int.TryParse(depthMatch.Groups[1].Value, out int depthValue))
                        depth = depthValue;
                }
                else if (line.StartsWith("bestmove ", StringComparison.OrdinalIgnoreCase))
                {
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                        bestMove = parts[1];

                    return new PikafishSearchScore(centipawns, matePly, depth, bestMove);
                }
            }
        }

        private static string ResolveWorkingDirectory(string enginePath)
        {
            string executableDirectory = Path.GetDirectoryName(enginePath) ?? Environment.CurrentDirectory;
            string? parentDirectory = Directory.GetParent(executableDirectory)?.FullName;
            if (parentDirectory != null && File.Exists(Path.Combine(parentDirectory, "pikafish.nnue")))
                return parentDirectory;

            return executableDirectory;
        }

        private async Task WaitReadyAsync(CancellationToken cancellationToken)
        {
            SendLine("isready");
            await WaitForLineAsync(line => line == "readyok", TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        }

        private void SendLine(string command)
        {
            if (_process.HasExited)
                throw new InvalidOperationException("Pikafish process has exited.");

            _process.StandardInput.WriteLine(command);
            _process.StandardInput.Flush();
        }

        private void ClearPendingLines()
        {
            while (_lines.TryDequeue(out _))
            {
            }
        }

        private async Task<string> WaitForLineAsync(Func<string, bool> predicate, TimeSpan timeout, CancellationToken cancellationToken)
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);

            while (true)
            {
                string line = await WaitForAnyLineAsync(linkedCts.Token).ConfigureAwait(false);
                if (predicate(line))
                    return line;
            }
        }

        private async Task<string> WaitForAnyLineAsync(CancellationToken cancellationToken)
        {
            while (true)
            {
                if (_lines.TryDequeue(out string? line))
                    return line;

                await _lineSignal.WaitAsync(cancellationToken).ConfigureAwait(false);
            }
        }

        private async Task ReadLoopAsync(StreamReader reader, CancellationToken cancellationToken)
        {
            try
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    string? line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
                    if (line == null)
                        break;

                    line = line.Trim();
                    if (line.Length == 0)
                        continue;

                    _lines.Enqueue(line);
                    _lineSignal.Release();
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
            }
            catch (ObjectDisposedException)
            {
            }
        }

        public void Dispose()
        {
            try
            {
                if (!_process.HasExited)
                {
                    try
                    {
                        SendLine("quit");
                    }
                    catch
                    {
                    }

                    if (!_process.WaitForExit(1000))
                        _process.Kill(entireProcessTree: true);
                }

                Task.WaitAll(new[] { _stdoutTask, _stderrTask }, 1000);
            }
            catch
            {
            }

            if (!_stdoutTask.IsCompleted || !_stderrTask.IsCompleted)
                _readerCts.Cancel();

            _readerCts.Dispose();
            _lineSignal.Dispose();
            _process.Dispose();
        }
    }
}
