using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;

namespace ChineseChessAI.Play
{
    public sealed class PikafishEngineClient : IDisposable
    {
        private readonly Process _process;
        private readonly ConcurrentQueue<string> _lines = new();
        private readonly SemaphoreSlim _lineSignal = new(0);
        private readonly CancellationTokenSource _readerCts = new();
        private readonly Task _stdoutTask;
        private readonly Task _stderrTask;

        public string EnginePath { get; }

        public PikafishEngineClient(string enginePath)
        {
            EnginePath = enginePath;
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

        private static string ResolveWorkingDirectory(string enginePath)
        {
            string executableDirectory = Path.GetDirectoryName(enginePath) ?? Environment.CurrentDirectory;
            string? parentDirectory = Directory.GetParent(executableDirectory)?.FullName;
            if (parentDirectory != null && File.Exists(Path.Combine(parentDirectory, "pikafish.nnue")))
                return parentDirectory;

            return executableDirectory;
        }

        public async Task InitializeAsync(CancellationToken cancellationToken)
        {
            SendLine("uci");
            await WaitForLineAsync(line => line == "uciok" || line == "ucciok", TimeSpan.FromSeconds(10), cancellationToken);
            await WaitReadyAsync(cancellationToken);
        }

        public async Task NewGameAsync(CancellationToken cancellationToken)
        {
            SendLine("ucinewgame");
            await WaitReadyAsync(cancellationToken);
        }

        public async Task<string> GetBestMoveAsync(
            IReadOnlyList<string> ucciHistory,
            int depth,
            int moveTimeMs,
            CancellationToken cancellationToken)
        {
            string moves = ucciHistory.Count == 0 ? string.Empty : " moves " + string.Join(' ', ucciHistory);
            SendLine("position startpos" + moves);

            if (moveTimeMs > 0)
                SendLine($"go movetime {moveTimeMs}");
            else
                SendLine($"go depth {Math.Max(1, depth)}");

            using var cancelRegistration = cancellationToken.Register(() =>
            {
                try
                {
                    SendLine("stop");
                }
                catch
                {
                    // Process may already be gone.
                }
            });

            string line = await WaitForLineAsync(
                text => text.StartsWith("bestmove ", StringComparison.OrdinalIgnoreCase),
                TimeSpan.FromSeconds(Math.Max(15, moveTimeMs / 1000 + 10)),
                cancellationToken);

            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2 || parts[1] == "0000" || parts[1] == "(none)")
                throw new InvalidOperationException($"Pikafish returned no move: {line}");

            return parts[1];
        }

        private async Task WaitReadyAsync(CancellationToken cancellationToken)
        {
            SendLine("isready");
            await WaitForLineAsync(line => line == "readyok", TimeSpan.FromSeconds(10), cancellationToken);
        }

        private void SendLine(string command)
        {
            if (_process.HasExited)
                throw new InvalidOperationException("Pikafish process has exited.");

            _process.StandardInput.WriteLine(command);
            _process.StandardInput.Flush();
        }

        private async Task<string> WaitForLineAsync(
            Func<string, bool> predicate,
            TimeSpan timeout,
            CancellationToken cancellationToken)
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);

            while (true)
            {
                while (_lines.TryDequeue(out string? line))
                {
                    if (predicate(line))
                        return line;
                }

                try
                {
                    await _lineSignal.WaitAsync(linkedCts.Token);
                }
                catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
                {
                    throw new TimeoutException("Timed out waiting for Pikafish response.");
                }
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
                // Expected during engine shutdown or engine replacement.
            }
            catch (ObjectDisposedException)
            {
                // Expected if the process streams are disposed during shutdown.
            }
        }

        public void Dispose()
        {
            _readerCts.Cancel();
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
                        // Ignore shutdown race.
                    }

                    if (!_process.WaitForExit(1000))
                        _process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // Ignore disposal failures.
            }

            _readerCts.Dispose();
            _lineSignal.Dispose();
            _process.Dispose();
        }
    }
}
