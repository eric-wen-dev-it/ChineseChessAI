using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Utils;
using System.Collections.Concurrent;
using System.IO;
using System.Text;
using System.Text.Json;
using TorchSharp;

namespace ChineseChessAI.Training
{
    public class PersistentAgent : IDisposable
    {
        public CChessNet Model
        {
            get;
        }
        public Trainer Trainer { get; private set; } = null!;

        public PersistentAgent()
        {
            Model = new CChessNet();
            // 【关键】：Trainer 不在构造函数中创建；调用方必须先完成 model.load()，
            // 然后调用 CompleteInit()，确保 Adam 优化器捕获的参数引用始终有效。
        }

        // 在 model.load() 和所有 .to() 调用完成后调用此方法
        internal void CompleteInit()
        {
            if (torch.cuda.is_available())
                Model.to(DeviceType.CUDA);
            Trainer = new Trainer(Model);
        }

        public void Dispose()
        {
            Trainer?.Dispose();
            Model?.Dispose();
        }
    }

    public class TrainingOrchestrator
    {
        public event Action<string>? OnLog;
        public event Action<List<Move>, int, int, string>? OnReplayRequested; // 增加结果参数
        public event Action<List<Move>, Move, string>? OnAuditFailureRequested; // 审计失败演示事件
        public event Action? OnTrainingStopped;
        public event Action<string>? OnError;

        private volatile bool _isTraining = false;
        public bool IsTraining => _isTraining;
        public ReplayBuffer MasterBuffer
        {
            get; private set;
        }
        public ReplayBuffer LeagueBuffer
        {
            get; private set;
        }

        private LeagueManager _leagueManager;
        private static readonly object _gpuTrainingLock = new object();
        private readonly SemaphoreSlim _maintenanceLock = new SemaphoreSlim(1, 1);
        private static readonly ConcurrentDictionary<string, object> _fileLocks = new();
        private static readonly object _runtimeLogLock = new object();
        private const int LoadedAgentCacheLimit = 12;
        private readonly object _inFlightGamesLock = new object();
        private static readonly TimeSpan LeagueGameTimeout = TimeSpan.FromMinutes(30);
        private static readonly TimeSpan LeagueDrainWaitTimeout = LeagueGameTimeout + TimeSpan.FromMinutes(3);
        private static readonly TimeSpan DrainProgressLogInterval = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan WatchdogCheckInterval = TimeSpan.FromMinutes(10);
        private static readonly TimeSpan WatchdogStaleLogThreshold = TimeSpan.FromMinutes(30);
        private static readonly TimeSpan LeagueShutdownWaitTimeout = TimeSpan.FromMinutes(1);
        private static readonly string _runtimeLogPath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "runtime.log");
        private static readonly string _leagueTimeoutRecordsDir = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "league_timeout_records");
        private const int MaxLeagueTrainingGames = 5000;
        private int _inFlightGameCount = 0;
        private TaskCompletionSource<bool> _gamesDrainedTcs = CreateCompletedTcs();
        private readonly ConcurrentDictionary<int, ActiveLeagueGame> _activeLeagueGames = new();

        private readonly ConcurrentDictionary<int, Lazy<PersistentAgent>> _agentPool = new();
        private readonly ConcurrentDictionary<int, long> _agentLastTouchedUtcTicks = new();
        private readonly ConcurrentDictionary<int, byte> _reservedAgentIds = new();
        private readonly ConcurrentDictionary<int, SemaphoreSlim> _agentActiveLocks = new();
        private SemaphoreSlim GetAgentActiveLock(int id) => _agentActiveLocks.GetOrAdd(id, _ => new SemaphoreSlim(1, 1));
        private object GetFileLock(string path) => _fileLocks.GetOrAdd(path, _ => new object());

        private CancellationTokenSource? _cts;
        private Task? _currentTrainingTask;
        private Task? _backgroundLoadTask;
        private CancellationTokenSource? _watchdogCts;
        private Task? _watchdogTask;
        private long _lastLogUtcTicks = DateTimeOffset.UtcNow.UtcTicks;
        private int _watchdogRestartRequested;
        private volatile bool _skipAgentDisposeOnNextStart;

        private sealed record LeagueRunOptions(
            int PopulationSize,
            int MaxMoves,
            int ExploreMoves,
            float MaterialBias,
            int PopulationRefreshInterval,
            int? MaxPopulationRefreshCycles);

        private sealed record ActiveLeagueGame(
            int GameId,
            int AgentA,
            int AgentB,
            DateTimeOffset StartedAt);

        private static TaskCompletionSource<bool> CreateCompletedTcs()
        {
            var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            tcs.TrySetResult(true);
            return tcs;
        }

        private static TaskCompletionSource<bool> CreatePendingTcs() =>
            new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

        private void TouchAgent(int agentId)
        {
            _agentLastTouchedUtcTicks[agentId] = DateTime.UtcNow.Ticks;
        }

        private void TrimIdleAgentPool(int maxLoaded = LoadedAgentCacheLimit)
        {
            if (_agentPool.Count <= maxLoaded)
                return;

            var reserved = GetReservedAgentIdsSnapshot();
            var candidates = _agentPool
                .Where(e => e.Value.IsValueCreated && !reserved.Contains(e.Key))
                .Select(e => (Id: e.Key, Tick: _agentLastTouchedUtcTicks.TryGetValue(e.Key, out long tick) ? tick : 0))
                .OrderBy(e => e.Tick)
                .ToList();

            foreach (var candidate in candidates)
            {
                if (_agentPool.Count <= maxLoaded)
                    break;

                var agentLock = GetAgentActiveLock(candidate.Id);
                if (!agentLock.Wait(0))
                    continue;

                try
                {
                    if (_reservedAgentIds.ContainsKey(candidate.Id))
                        continue;

                    if (_agentPool.TryRemove(candidate.Id, out var lazyAgent) && lazyAgent.IsValueCreated)
                    {
                        lazyAgent.Value.Dispose();
                        _agentLastTouchedUtcTicks.TryRemove(candidate.Id, out _);
                    }
                }
                finally
                {
                    agentLock.Release();
                }
            }
        }

        private void ResetInFlightGameTracking()
        {
            lock (_inFlightGamesLock)
            {
                _inFlightGameCount = 0;
                _gamesDrainedTcs = CreateCompletedTcs();
            }

            _activeLeagueGames.Clear();
        }

        private void MarkGameStarted(int gameId, int agentIdA, int agentIdB)
        {
            _activeLeagueGames[gameId] = new ActiveLeagueGame(gameId, agentIdA, agentIdB, DateTimeOffset.Now);

            lock (_inFlightGamesLock)
            {
                if (_inFlightGameCount == 0)
                {
                    _gamesDrainedTcs = CreatePendingTcs();
                }

                _inFlightGameCount++;
            }
        }

        private void MarkGameFinished(int gameId)
        {
            TaskCompletionSource<bool>? drained = null;
            _activeLeagueGames.TryRemove(gameId, out _);

            lock (_inFlightGamesLock)
            {
                if (_inFlightGameCount <= 0)
                {
                    return;
                }

                _inFlightGameCount--;
                if (_inFlightGameCount == 0)
                {
                    drained = _gamesDrainedTcs;
                }
            }

            drained?.TrySetResult(true);
        }

        private int GetInFlightGameCount()
        {
            lock (_inFlightGamesLock)
            {
                return _inFlightGameCount;
            }
        }

        private string GetActiveLeagueGamesSummary()
        {
            var activeGames = _activeLeagueGames.Values
                .OrderBy(g => g.StartedAt)
                .ToList();

            if (activeGames.Count == 0)
            {
                return "活跃对局明细: 无";
            }

            DateTimeOffset now = DateTimeOffset.Now;
            var entries = activeGames.Select(g =>
                $"#{g.GameId} Agent_{g.AgentA} vs Agent_{g.AgentB}, 已运行 {(now - g.StartedAt).TotalMinutes:F1} 分钟, 开始 {g.StartedAt:HH:mm:ss}");
            return "活跃对局明细: " + string.Join(" | ", entries);
        }

        private HashSet<int> GetReservedAgentIdsSnapshot()
        {
            return _reservedAgentIds.Keys.ToHashSet();
        }

        private bool TryReserveAgents(int agentIdA, int agentIdB)
        {
            if (!_reservedAgentIds.TryAdd(agentIdA, 0))
            {
                return false;
            }

            if (_reservedAgentIds.TryAdd(agentIdB, 0))
            {
                return true;
            }

            _reservedAgentIds.TryRemove(agentIdA, out _);
            return false;
        }

        private void ReleaseReservedAgents(int agentIdA, int agentIdB)
        {
            _reservedAgentIds.TryRemove(agentIdA, out _);
            _reservedAgentIds.TryRemove(agentIdB, out _);
        }

        private Task WaitForInFlightGamesToDrainAsync(CancellationToken token)
        {
            Task waitTask;
            int inFlightGames;

            lock (_inFlightGamesLock)
            {
                inFlightGames = _inFlightGameCount;
                waitTask = _gamesDrainedTcs.Task;
            }

            if (inFlightGames == 0)
            {
                return Task.CompletedTask;
            }

            return waitTask.WaitAsync(token);
        }

        private async Task WaitForInFlightGamesToDrainWithProgressAsync(string context, CancellationToken token)
        {
            DateTimeOffset startedAt = DateTimeOffset.Now;

            while (true)
            {
                int inFlightGames = GetInFlightGameCount();
                if (inFlightGames == 0)
                {
                    return;
                }

                TimeSpan elapsed = DateTimeOffset.Now - startedAt;
                if (elapsed > LeagueDrainWaitTimeout)
                {
                    string message = $"[{context}] 等待对局收束超时：仍有 {inFlightGames} 个对局未结束，已等待 {elapsed.TotalMinutes:F1} 分钟。{GetActiveLeagueGamesSummary()}";
                    Log(message);
                    throw new TimeoutException(message);
                }

                Log($"[{context}] 等待中：当前仍有 {inFlightGames} 个对局在进行，已等待 {elapsed.TotalMinutes:F1} 分钟。{GetActiveLeagueGamesSummary()}");

                Task drainTask = WaitForInFlightGamesToDrainAsync(token);
                Task delayTask = Task.Delay(DrainProgressLogInterval, token);
                Task completedTask = await Task.WhenAny(drainTask, delayTask).ConfigureAwait(false);

                if (completedTask == drainTask)
                {
                    await drainTask.ConfigureAwait(false);
                    return;
                }

                await delayTask.ConfigureAwait(false);
            }
        }

        private async Task WaitForGameTasksToSettleAsync(IEnumerable<Task> gameTasks, CancellationToken token)
        {
            Task[] activeTasks = gameTasks.Where(t => t != null && !t.IsCompleted).ToArray();
            if (activeTasks.Length == 0)
            {
                return;
            }

            Task allGamesTask = Task.WhenAll(activeTasks);
            Task timeoutTask = Task.Delay(LeagueShutdownWaitTimeout, CancellationToken.None);
            Task completedTask = await Task.WhenAny(allGamesTask, timeoutTask).ConfigureAwait(false);
            if (completedTask == allGamesTask)
            {
                await allGamesTask.ConfigureAwait(false);
                return;
            }

            Log($"[联赛关闭] 等待对局任务结束超过 {LeagueShutdownWaitTimeout.TotalMinutes:F0} 分钟，仍有 {activeTasks.Count(t => !t.IsCompleted)} 个任务未退出。");
        }

        private DateTimeOffset GetLastObservedLogUtc()
        {
            long inMemoryTicks = Volatile.Read(ref _lastLogUtcTicks);
            DateTimeOffset last = new DateTimeOffset(inMemoryTicks, TimeSpan.Zero);

            try
            {
                if (File.Exists(_runtimeLogPath))
                {
                    DateTime fileWriteTimeUtc = File.GetLastWriteTimeUtc(_runtimeLogPath);
                    if (fileWriteTimeUtc.Ticks > last.UtcTicks)
                    {
                        last = new DateTimeOffset(fileWriteTimeUtc, TimeSpan.Zero);
                    }
                }
            }
            catch
            {
            }

            return last;
        }

        private void StartLeagueWatchdog(LeagueRunOptions options, CancellationToken runToken)
        {
            _watchdogCts?.Cancel();
            _watchdogCts?.Dispose();
            _watchdogCts = CancellationTokenSource.CreateLinkedTokenSource(runToken);
            _watchdogRestartRequested = 0;
            Volatile.Write(ref _lastLogUtcTicks, DateTimeOffset.UtcNow.UtcTicks);

            var watchdogToken = _watchdogCts.Token;
            _watchdogTask = Task.Run(async () =>
            {
                while (!watchdogToken.IsCancellationRequested)
                {
                    try
                    {
                        await Task.Delay(WatchdogCheckInterval, watchdogToken).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }

                    DateTimeOffset lastLogUtc = GetLastObservedLogUtc();
                    TimeSpan quietFor = DateTimeOffset.UtcNow - lastLogUtc;
                    if (quietFor < WatchdogStaleLogThreshold)
                    {
                        continue;
                    }

                    await RestartLeagueFromWatchdogAsync(options, quietFor, lastLogUtc).ConfigureAwait(false);
                    break;
                }
            });
        }

        private async Task RestartLeagueFromWatchdogAsync(LeagueRunOptions options, TimeSpan quietFor, DateTimeOffset lastLogUtc)
        {
            if (Interlocked.CompareExchange(ref _watchdogRestartRequested, 1, 0) != 0)
            {
                return;
            }

            Log("[Watchdog] 检测到日志长时间无输出，准备写入状态并重启联赛。");
            Log($"[Watchdog] 最后日志时间 UTC: {lastLogUtc:yyyy-MM-dd HH:mm:ss}，静默 {quietFor.TotalMinutes:F1} 分钟。");
            LogLeagueStateSnapshot("[Watchdog]");

            Task? previousTrainingTask = _currentTrainingTask;
            StopTraining();

            if (previousTrainingTask != null)
            {
                Task completedTask = await Task.WhenAny(previousTrainingTask, Task.Delay(LeagueShutdownWaitTimeout)).ConfigureAwait(false);
                if (completedTask != previousTrainingTask)
                {
                    Log($"[Watchdog] 旧联赛在 {LeagueShutdownWaitTimeout.TotalMinutes:F0} 分钟内未完全退出；将保留旧 agent 对象引用并启动新联赛，避免释放仍可能被旧任务使用的模型。");
                    _skipAgentDisposeOnNextStart = true;
                    _currentTrainingTask = null;
                    _backgroundLoadTask = null;
                }
            }

            Log("[Watchdog] 正在按原参数重启联赛。");
            await StartLeagueTrainingAsync(
                options.PopulationSize,
                options.MaxMoves,
                options.ExploreMoves,
                options.MaterialBias,
                options.PopulationRefreshInterval,
                options.MaxPopulationRefreshCycles).ConfigureAwait(false);
        }

        private void LogLeagueStateSnapshot(string prefix)
        {
            Log($"{prefix} 状态快照：IsTraining={IsTraining}, InFlight={GetInFlightGameCount()}, ReservedAgents={_reservedAgentIds.Count}, LoadedAgents={_agentPool.Count}");
            Log($"{prefix} 任务状态：TrainingTask={_currentTrainingTask?.Status.ToString() ?? "null"}, BackgroundLoad={_backgroundLoadTask?.Status.ToString() ?? "null"}, Canceled={_cts?.IsCancellationRequested ?? false}");
            Log($"{prefix} {GetActiveLeagueGamesSummary()}");

            var reserved = GetReservedAgentIdsSnapshot();
            if (reserved.Count > 0)
            {
                Log($"{prefix} 保留中的 Agent: {string.Join(",", reserved.OrderBy(x => x))}");
            }
        }

        public TrainingOrchestrator()
        {
            MasterBuffer = new ReplayBuffer(5000000, "data/master_data");
            LeagueBuffer = new ReplayBuffer(1000000, "data/league_data");

            MasterBuffer.OnSaveError += msg => { Log(msg); OnError?.Invoke(msg); };
            LeagueBuffer.OnSaveError += msg => { Log(msg); OnError?.Invoke(msg); };
        }

        public void StopTraining()
        {
            _isTraining = false;
            _cts?.Cancel();
        }

        public async Task StartLeagueTrainingAsync(
            int populationSize = 50,
            int maxMoves = 150,
            int exploreMoves = 40,
            float materialBias = 0.1f,
            int populationRefreshInterval = 0,
            int? maxPopulationRefreshCycles = null)
        {
            if (populationSize > 100)
                throw new ArgumentException("出于内存限制与并发安全考量，联赛人口数量不能超过 100。", nameof(populationSize));
            if (populationSize < 2)
                throw new ArgumentException("联赛人口数量必须大于等于 2。", nameof(populationSize));

            if (populationRefreshInterval <= 0)
                populationRefreshInterval = populationSize * 8;
            populationRefreshInterval = Math.Max(populationRefreshInterval, populationSize * 4);
            if (maxPopulationRefreshCycles.HasValue && maxPopulationRefreshCycles.Value <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxPopulationRefreshCycles));
            var runOptions = new LeagueRunOptions(
                populationSize,
                maxMoves,
                exploreMoves,
                materialBias,
                populationRefreshInterval,
                maxPopulationRefreshCycles);

            if (IsTraining)
                return;
            if (_currentTrainingTask != null && !_currentTrainingTask.IsCompleted)
            {
                try
                {
                    await _currentTrainingTask;
                }
                catch { }
            }
            if (_backgroundLoadTask != null && !_backgroundLoadTask.IsCompleted)
            {
                try
                {
                    await _backgroundLoadTask;
                }
                catch { }
            }

            _isTraining = true;
            _cts = new CancellationTokenSource();
            var runCts = _cts;
            StartLeagueWatchdog(runOptions, runCts.Token);
            var runWatchdogCts = _watchdogCts;

            _leagueManager = new LeagueManager(populationSize);
            if (_skipAgentDisposeOnNextStart)
            {
                Log("[Watchdog] 跳过本次启动前的旧 agent dispose；旧联赛任务可能仍在退出。");
                _skipAgentDisposeOnNextStart = false;
            }
            else
            {
                foreach (var lazyAgent in _agentPool.Values)
                    if (lazyAgent.IsValueCreated)
                        lazyAgent.Value.Dispose();
            }

            _agentPool.Clear();
            _agentLastTouchedUtcTicks.Clear();
            _reservedAgentIds.Clear();
            _agentActiveLocks.Clear();
            ResetInFlightGameTracking();

            DateTime startTime = DateTime.Now;
            MasterBuffer.Clear();
            LeagueBuffer.Clear();

            _currentTrainingTask = Task.Run(async () =>
            {
                try
                {
                    Log($"=== 万王之王：{populationSize} 智能体联赛启动 ===");

                    // 将数据装载放入独立的后台任务，不阻塞联赛和对局的立即启动
                    _backgroundLoadTask = Task.Run(async () =>
                    {
                        try
                        {
                            Log("[后台任务] 正在静默装载大师数据与历史联赛数据...");

                            var masterTask = MasterBuffer.LoadOldSamplesAsync(int.MaxValue, logAction: Log, onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r), cancellationToken: runCts.Token, cutoffTime: startTime);
                            var leagueTask = LeagueBuffer.LoadOldSamplesAsync(int.MaxValue, logAction: Log, onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r), cancellationToken: runCts.Token, cutoffTime: startTime);

                            await Task.WhenAll(masterTask, leagueTask);

                            var (masterSamples, masterGames) = await masterTask;
                            var (leagueSamples, leagueGames) = await leagueTask;

                            Log($"[后台装载完成] 大师数据: {masterGames} 局 ({masterSamples} 条) | 联赛数据: {leagueGames} 局 ({leagueSamples} 条)");
                        }
                        catch (Exception ex) { Log($"[后台装载异常] {ex.Message}"); }
                    }, runCts.Token);

                    const int maxParallelGames = 4;
                    int gameCounter = 0;
                    int completedGameCounter = 0;
                    int nextLogAt = 10;
                    int nextTrainAt = 20;
                    int nextEvolutionAt = populationRefreshInterval;
                    int completedRefreshCycles = 0;

                    var semaphore = new SemaphoreSlim(maxParallelGames);
                    var gameTasks = new System.Collections.Concurrent.ConcurrentQueue<Task>();

                    while (IsTraining)
                    {
                        try
                        {
                            await semaphore.WaitAsync(runCts.Token);
                        }
                        catch (OperationCanceledException)
                        {
                            break;
                        }

                        if (!IsTraining || runCts.Token.IsCancellationRequested)
                        {
                            semaphore.Release();
                            break;
                        }

                        if (completedGameCounter >= nextTrainAt)
                        {
                            nextTrainAt += 20;
                            semaphore.Release();
                            await PerformDiverseTrainingAsync(runCts.Token);
                            if (runCts.Token.IsCancellationRequested || !IsTraining)
                            {
                                break;
                            }

                            continue;
                        }

                        if (completedGameCounter >= nextEvolutionAt)
                        {
                            nextEvolutionAt += populationRefreshInterval;
                            semaphore.Release();
                            await PerformPopulationRefreshAsync(runCts.Token);
                            if (runCts.Token.IsCancellationRequested || !IsTraining)
                            {
                                break;
                            }

                            completedRefreshCycles++;
                            if (maxPopulationRefreshCycles.HasValue && completedRefreshCycles >= maxPopulationRefreshCycles.Value)
                            {
                                Log($"[PopulationRefresh] Completed {completedRefreshCycles}/{maxPopulationRefreshCycles.Value} cycle(s); stopping league run.");
                                StopTraining();
                                break;
                            }

                            continue;
                        }

                        var reservedAgentIds = GetReservedAgentIdsSnapshot();
                        if (!_leagueManager.TryPickMatch(reservedAgentIds, out var agentMetaA, out var agentMetaB))
                        {
                            semaphore.Release();
                            try
                            {
                                await Task.Delay(200, runCts.Token);
                            }
                            catch (OperationCanceledException)
                            {
                            }

                            continue;
                        }

                        if (!TryReserveAgents(agentMetaA.Id, agentMetaB.Id))
                        {
                            semaphore.Release();
                            try
                            {
                                await Task.Delay(50, runCts.Token);
                            }
                            catch (OperationCanceledException)
                            {
                            }

                            continue;
                        }

                        int currentId = Interlocked.Increment(ref gameCounter);
                        MarkGameStarted(currentId, agentMetaA.Id, agentMetaB.Id);
                        int currentMaxMoves = (int)(maxMoves * (0.7 + Random.Shared.NextDouble() * 0.8));
                        Task gameTask;
                        try
                        {
                            gameTask = Task.Run(async () =>
                            {
                                try
                                {
                                    var (firstMeta, secondMeta) = agentMetaA.Id < agentMetaB.Id ? (agentMetaA, agentMetaB) : (agentMetaB, agentMetaA);
                                    var lockFirst = GetAgentActiveLock(firstMeta.Id);
                                    var lockSecond = GetAgentActiveLock(secondMeta.Id);

                                    await lockFirst.WaitAsync(runCts.Token);
                                    try
                                    {
                                        await lockSecond.WaitAsync(runCts.Token);
                                        try
                                        {
                                            var pAgentA = GetOrAddAgent(agentMetaA);
                                            var pAgentB = GetOrAddAgent(agentMetaB);

                                            using var engineA = new MCTSEngine(pAgentA.Model, batchSize: 16, cPuct: agentMetaA.Cpuct);
                                            using var engineB = new MCTSEngine(pAgentB.Model, batchSize: 16, cPuct: agentMetaB.Cpuct);

                                            var selfPlay = new SelfPlay(engineA, engineB, currentMaxMoves, exploreMoves, materialBias,
                                                                        lowTempA: agentMetaA.Temperature, lowTempB: agentMetaB.Temperature,
                                                                        simsA: agentMetaA.MctsSimulations, simsB: agentMetaB.MctsSimulations);

                                            Log($"[对局 #{currentId} 开始] Agent_{agentMetaA.Id}(ELO:{agentMetaA.Elo:F0} DNA:S{agentMetaA.MctsSimulations}/C{agentMetaA.Cpuct:F1}/T{agentMetaA.Temperature:F1}) " +
                                                $"VS Agent_{agentMetaB.Id}(ELO:{agentMetaB.Elo:F0} DNA:S{agentMetaB.MctsSimulations}/C{agentMetaB.Cpuct:F1}/T{agentMetaB.Temperature:F1})");

                                            bool aIsRed = Random.Shared.Next(2) == 0;
                                            using var gameTimeoutCts = CancellationTokenSource.CreateLinkedTokenSource(runCts.Token);
                                            gameTimeoutCts.CancelAfter(LeagueGameTimeout);
                                            using var timeoutNoticeCts = CancellationTokenSource.CreateLinkedTokenSource(runCts.Token);
                                            _ = LogGameTimeoutRequestAsync(
                                                currentId,
                                                agentMetaA.Id,
                                                agentMetaB.Id,
                                                gameTimeoutCts.Token,
                                                timeoutNoticeCts.Token);
                                            var result = await selfPlay.RunGameAsync(aIsRed, null, gameTimeoutCts.Token);
                                            timeoutNoticeCts.Cancel();
                                            bool hitGameTimeout = gameTimeoutCts.IsCancellationRequested && !runCts.Token.IsCancellationRequested;

                                            if (result.IsSuccess)
                                            {
                                                float resA = result.ResultStr == "平局" ? 0
                                                    : (result.ResultStr == (aIsRed ? "红胜" : "黑胜") ? 1.0f : -1.0f);

                                                // 【修复 P1】：捕获赛前 ELO 以确保更新公平
                                                double eloABefore = agentMetaA.Elo;
                                                double eloBBefore = agentMetaB.Elo;

                                                lock (_leagueManager)
                                                {
                                                    _leagueManager.UpdateResult(agentMetaA.Id, resA, eloBBefore);
                                                    _leagueManager.UpdateResult(agentMetaB.Id, -resA, eloABefore);
                                                }

                                                var combinedExamples = new List<TrainingExample>(result.ExamplesA.Count + result.ExamplesB.Count);
                                                combinedExamples.AddRange(result.ExamplesA);
                                                combinedExamples.AddRange(result.ExamplesB);

                                                if (combinedExamples.Count > 0)
                                                {
                                                    var moveHistoryUcci = result.MoveHistory.Select(NotationConverter.MoveToUcci).ToList();
                                                    LeagueBuffer.AddGame(new MasterGameData(combinedExamples, moveHistoryUcci)
                                                    {
                                                        StartedAt = result.StartedAt,
                                                        EndedAt = result.EndedAt,
                                                        Elapsed = result.Elapsed,
                                                        Result = result.ResultStr,
                                                        EndReason = result.EndReason,
                                                        MoveCount = result.MoveCount,
                                                        GameId = currentId
                                                    });
                                                }

                                                Log($"[对局 #{currentId} 结束] Agent_{agentMetaA.Id}(ELO:{agentMetaA.Elo:F0}) VS Agent_{agentMetaB.Id}(ELO:{agentMetaB.Elo:F0}) | {result.ResultStr} | {result.MoveCount}步");

                                                OnReplayRequested?.Invoke(result.MoveHistory, currentMaxMoves, currentId, result.ResultStr);
                                            }
                                            else if (hitGameTimeout)
                                            {
                                                string? savedRecordPath = SaveTimedOutLeagueRecord(currentId, agentMetaA, agentMetaB, result);
                                                string savedSuffix = savedRecordPath == null ? string.Empty : $" | 记录: {Path.GetFileName(savedRecordPath)}";
                                                Log($"[对局 #{currentId} 超时] Agent_{agentMetaA.Id} VS Agent_{agentMetaB.Id} | 超过 {LeagueGameTimeout.TotalMinutes:F0} 分钟终止 | 已走 {result.MoveCount} 步{savedSuffix}");

                                                if (result.MoveHistory.Count > 0)
                                                {
                                                    OnReplayRequested?.Invoke(result.MoveHistory, currentMaxMoves, currentId, "超时终结");
                                                }
                                            }
                                            else if (result.EndReason != "训练被强制终止")
                                            {
                                                throw new Exception($"对弈失败 - {result.EndReason}");
                                            }
                                        }
                                        finally
                                        {
                                            Interlocked.Increment(ref completedGameCounter);
                                            lockSecond.Release();
                                        }
                                    }
                                    finally { lockFirst.Release(); }
                                }
                                catch (Exception ex)
                                {
                                    Log($"[对局异常] {ex.Message}");
                                    Log($"[对局异常-堆栈] {ex}");
                                }
                                finally
                                {
                                    ReleaseReservedAgents(agentMetaA.Id, agentMetaB.Id);
                                    MarkGameFinished(currentId);
                                    semaphore.Release();
                                    TrimIdleAgentPool();
                                }
                            });
                        }
                        catch
                        {
                            ReleaseReservedAgents(agentMetaA.Id, agentMetaB.Id);
                            MarkGameFinished(currentId);
                            semaphore.Release();
                            throw;
                        }

                        gameTasks.Enqueue(gameTask);
                        while (gameTasks.Count > 20 && gameTasks.TryPeek(out var first) && first.IsCompleted)
                        {
                            gameTasks.TryDequeue(out _);
                        }

                        if (completedGameCounter >= nextLogAt)
                        {
                            nextLogAt += 10;
                            _leagueManager.SaveMetadata();
                            var top = _leagueManager.GetTopAgents(5);
                            Log("--- [当前排名 Top 5] ---");
                            foreach (var t in top)
                                Log($"ID:{t.Id} ELO:{t.Elo:F0} 胜率:{(t.Wins * 100.0 / Math.Max(1, t.GamesPlayed)):F1}%");
                            // 【优化 P3 #8】：移除阻塞式 GC.Collect()，交给 .NET 自动管理
                        }

                    }

                    try
                    {
                        await WaitForGameTasksToSettleAsync(gameTasks, runCts.Token);
                    }
                    catch { }
                }
                catch (Exception ex) { OnError?.Invoke($"[系统故障] {ex.Message}"); }
                finally
                {
                    if (ReferenceEquals(_watchdogCts, runWatchdogCts))
                    {
                        _watchdogCts?.Cancel();
                    }
                    if (_backgroundLoadTask != null && !_backgroundLoadTask.IsCompleted)
                    {
                        try
                        {
                            await _backgroundLoadTask;
                        }
                        catch { }
                    }
                    _isTraining = false;
                    _leagueManager?.SaveMetadata();
                    OnTrainingStopped?.Invoke();
                }
            });
        }

        private PersistentAgent GetOrAddAgent(AgentMetadata meta)
        {
            PersistentAgent agent = _agentPool.GetOrAdd(meta.Id, id => new Lazy<PersistentAgent>(() =>
            {
                var pa = new PersistentAgent();
                // 【关键修复】：load() 必须在 CompleteInit()（即 Trainer/Adam 创建）之前完成。
                // TorchSharp 的 load() 会替换参数张量包装器对象；若 Adam 已创建，
                // 其持有的旧引用 handle 变为 IntPtr.Zero，下次 zero_grad() 即崩溃。
                lock (GetFileLock(meta.ModelPath))
                {
                    if (File.Exists(meta.ModelPath))
                    {
                        try
                        {
                            pa.Model.load(meta.ModelPath);
                        }
                        catch (EndOfStreamException ex)
                        {
                            string quarantinedPath = QuarantineCorruptModelFile(meta.ModelPath);
                            Log($"[模型损坏] Agent_{meta.Id} 模型文件已截断或损坏: {meta.ModelPath}");
                            Log($"[模型损坏] 已隔离到: {quarantinedPath}");
                            Log($"[模型损坏-堆栈] {ex}");
                        }
                    }
                }

                pa.CompleteInit(); // to(CUDA) + new Trainer(Model)，必须在 load() 之后
                return pa;
            }, LazyThreadSafetyMode.ExecutionAndPublication)).Value;
            TouchAgent(meta.Id);
            return agent;
        }

        private string QuarantineCorruptModelFile(string modelPath)
        {
            string directory = Path.GetDirectoryName(modelPath) ?? AppDomain.CurrentDomain.BaseDirectory;
            string fileName = Path.GetFileName(modelPath);
            string quarantinedPath = Path.Combine(
                directory,
                $"{Path.GetFileNameWithoutExtension(fileName)}.corrupt_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}{Path.GetExtension(fileName)}");

            try
            {
                File.Move(modelPath, quarantinedPath, overwrite: false);
            }
            catch (IOException)
            {
                quarantinedPath = Path.Combine(
                    directory,
                    $"{Path.GetFileNameWithoutExtension(fileName)}.corrupt_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}.bak");
                File.Copy(modelPath, quarantinedPath, overwrite: false);
                File.Delete(modelPath);
            }

            return quarantinedPath;
        }

        private async Task PerformPopulationRefreshAsync(CancellationToken token)
        {
            bool maintenanceLockHeld = false;
            var heldLocks = new List<SemaphoreSlim>();

            try
            {
                Log("[种群重组] 开始：等待当前对局与训练批次安全收束...");

                await _maintenanceLock.WaitAsync(token);
                maintenanceLockHeld = true;
                Log("[PopulationRefresh] Maintenance gate acquired; waiting for in-flight games to drain.");
                await WaitForInFlightGamesToDrainWithProgressAsync("PopulationRefresh", token);

                Log("[PopulationRefresh] Games drained; waiting for agent activity locks.");

                foreach (int agentId in _leagueManager.GetAllAgentIds())
                {
                    var agentLock = GetAgentActiveLock(agentId);
                    await agentLock.WaitAsync(token);
                    heldLocks.Add(agentLock);
                }

                lock (_gpuTrainingLock)
                {
                    FlushLoadedModelsToDisk();

                    int populationSize = _leagueManager.GetPopulationSize();
                    int eliteCount = Math.Clamp(populationSize / 10, 1, Math.Max(1, populationSize - 3));
                    int contenderKeepCount = Math.Clamp(populationSize * 3 / 10, 1, Math.Max(1, populationSize - eliteCount - 2));
                    int diverseKeepCount = Math.Clamp(populationSize / 5, 1, Math.Max(1, populationSize - eliteCount - contenderKeepCount - 1));
                    int parentPoolSize = Math.Clamp(Math.Min(10, Math.Max(4, populationSize / 5)), 1, populationSize);

                    int replacementCount = Math.Max(0, populationSize - eliteCount - contenderKeepCount - diverseKeepCount);
                    int immigrantCount = replacementCount > 0 ? Math.Clamp(Math.Max(1, replacementCount / 5), 1, replacementCount) : 0;

                    var refresh = _leagueManager.RefreshPopulation(
                        eliteCount,
                        contenderKeepCount,
                        diverseKeepCount,
                        parentPoolSize,
                        immigrantCount);

                    RefreshAgentPool(refresh.ReplacedAgentIds);
                    TrimIdleAgentPool();

                    if (refresh.Replaced > 0)
                    {
                        Log($"[种群重组] 完成：精英保留 {refresh.EliteKept}，竞争者保留 {refresh.ContenderKept}，多样性保留 {refresh.DiverseKept}，重建 {refresh.Replaced}（后代 {refresh.OffspringCreated}，移民 {refresh.ImmigrantsCreated}）。");
                        foreach (string line in refresh.PreviewLines)
                        {
                            Log($"[种群重组] {line}");
                        }
                    }
                    else
                    {
                        Log("[种群重组] 跳过：当前种群规模不足以执行安全重组。");
                    }
                }
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex)
            {
                Log($"[种群重组异常] {ex.Message}");
                Log($"[种群重组异常-堆栈] {ex}");
                OnError?.Invoke($"[种群重组异常] {ex.Message}");
            }
            finally
            {
                for (int i = heldLocks.Count - 1; i >= 0; i--)
                {
                    heldLocks[i].Release();
                }

                if (maintenanceLockHeld)
                {
                    _maintenanceLock.Release();
                }
            }
        }

        private void FlushLoadedModelsToDisk()
        {
            foreach (var agentEntry in _agentPool)
            {
                if (!agentEntry.Value.IsValueCreated)
                {
                    continue;
                }

                var meta = _leagueManager.GetAgentMeta(agentEntry.Key);
                if (meta == null)
                {
                    continue;
                }

                lock (GetFileLock(meta.ModelPath))
                {
                    ModelManager.SaveModel(agentEntry.Value.Value.Model, meta.ModelPath);
                }
            }
        }

        private string? SaveTimedOutLeagueRecord(int gameId, AgentMetadata agentMetaA, AgentMetadata agentMetaB, GameResult result)
        {
            if (result.MoveHistory.Count == 0)
            {
                return null;
            }

            try
            {
                Directory.CreateDirectory(_leagueTimeoutRecordsDir);

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string filePath = Path.Combine(_leagueTimeoutRecordsDir, $"timeout_game_{timestamp}_{gameId}_{Guid.NewGuid():N}.json");
                var moveHistoryUcci = result.MoveHistory.Select(NotationConverter.MoveToUcci).ToList();
                var record = new
                {
                    Examples = Array.Empty<TrainingExample>(),
                    MoveHistoryUcci = moveHistoryUcci,
                    Result = "超时终结",
                    result.EndReason,
                    result.MoveCount,
                    GameId = gameId,
                    CreatedAt = DateTime.Now,
                    AgentA = new
                    {
                        agentMetaA.Id,
                        agentMetaA.Elo,
                        agentMetaA.MctsSimulations,
                        agentMetaA.Cpuct,
                        agentMetaA.Temperature
                    },
                    AgentB = new
                    {
                        agentMetaB.Id,
                        agentMetaB.Elo,
                        agentMetaB.MctsSimulations,
                        agentMetaB.Cpuct,
                        agentMetaB.Temperature
                    }
                };

                lock (GetFileLock(filePath))
                {
                    File.WriteAllText(filePath, JsonSerializer.Serialize(record, new JsonSerializerOptions
                    {
                        WriteIndented = true
                    }));
                }

                return filePath;
            }
            catch (Exception ex)
            {
                Log($"[对局记录保存失败] #{gameId}: {ex.Message}");
                return null;
            }
        }

        private async Task LogGameTimeoutRequestAsync(
            int gameId,
            int agentIdA,
            int agentIdB,
            CancellationToken gameTimeoutToken,
            CancellationToken completionToken)
        {
            try
            {
                Task timeoutTask = Task.Delay(Timeout.InfiniteTimeSpan, gameTimeoutToken);
                Task completionTask = Task.Delay(Timeout.InfiniteTimeSpan, completionToken);
                Task completedTask = await Task.WhenAny(timeoutTask, completionTask).ConfigureAwait(false);
                if (completedTask == timeoutTask && gameTimeoutToken.IsCancellationRequested && !completionToken.IsCancellationRequested)
                {
                    Log($"[对局 #{gameId} 超时请求] Agent_{agentIdA} VS Agent_{agentIdB} | 已达到 {LeagueGameTimeout.TotalMinutes:F0} 分钟，已请求取消，等待对局任务退出。");
                }
            }
            catch
            {
            }
        }

        private void RefreshAgentPool(IEnumerable<int> replacedIds)
        {
            foreach (int agentId in replacedIds)
            {
                if (_agentPool.TryRemove(agentId, out var lazyAgent) && lazyAgent.IsValueCreated)
                {
                    lazyAgent.Value.Dispose();
                }

                _agentLastTouchedUtcTicks.TryRemove(agentId, out _);
            }
        }

        private async Task PerformDiverseTrainingAsync(CancellationToken token)
        {
            await Task.Run(() =>
            {
                bool maintenanceLockHeld = false;
                var heldLocks = new List<SemaphoreSlim>();
                try
                {
                    Log("[周期训练] 开始：等待当前对局安全收束...");

                    _maintenanceLock.Wait(token);
                    maintenanceLockHeld = true;
                    WaitForInFlightGamesToDrainWithProgressAsync("周期训练", token).GetAwaiter().GetResult();

                    foreach (int agentId in _leagueManager.GetAllAgentIds())
                    {
                        var agentLock = GetAgentActiveLock(agentId);
                        agentLock.Wait(token);
                        heldLocks.Add(agentLock);
                    }

                    var (leagueSamples, leagueGames, deletedLeagueGames) = LeagueBuffer
                        .RetainMostRecentGamesAsync(
                            MaxLeagueTrainingGames,
                            logAction: Log,
                            onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r),
                            cancellationToken: token)
                        .GetAwaiter()
                        .GetResult();

                    Log($"[周期训练] 联赛样本清理：保留最近 {leagueGames} 局（{leagueSamples} 条），删除 {deletedLeagueGames} 局旧对局。");

                    int trainedAgents = 0;
                    int skippedBusyAgents = 0;
                    int skippedUninitializedAgents = 0;
                    int skippedNoDataAgents = 0;
                    int totalSamples = 0;
                    float totalLoss = 0f;

                    int populationSize = _leagueManager.GetPopulationSize();
                    Log($"[周期训练] 开始：大师样本 {MasterBuffer.Count}，联赛样本 {LeagueBuffer.Count}，训练智能体 {populationSize}");

                    lock (_gpuTrainingLock)
                    {
                        const int batchSize = 192;
                        const int trainingEpochs = 2;
                        const float masterRatio = 0.3f;
                        const float leagueRatio = 0.7f;

                        foreach (int agentId in _leagueManager.GetAllAgentIds())
                        {
                            if (token.IsCancellationRequested)
                                return;

                            var meta = _leagueManager.GetAgentMeta(agentId);
                            if (meta == null)
                            {
                                continue;
                            }

                            var pa = GetOrAddAgent(meta);

                            var mixedBatch = new List<TrainingExample>();
                            if (MasterBuffer.Count > 0)
                                mixedBatch.AddRange(MasterBuffer.Sample((int)(batchSize * masterRatio)));
                            if (LeagueBuffer.Count > 0)
                                mixedBatch.AddRange(LeagueBuffer.Sample((int)(batchSize * leagueRatio)));

                            if (mixedBatch.Count > 0)
                            {
                                float loss = pa.Trainer.Train(mixedBatch, epochs: trainingEpochs);
                                trainedAgents++;
                                totalSamples += mixedBatch.Count * trainingEpochs;
                                totalLoss += loss;
                                lock (GetFileLock(meta.ModelPath))
                                {
                                    ModelManager.SaveModel(pa.Model, meta.ModelPath);
                                }
                            }
                            else
                            {
                                skippedNoDataAgents++;
                            }
                        }

                        TrimIdleAgentPool(populationSize);
                    }

                    if (!token.IsCancellationRequested)
                    {
                        if (trainedAgents > 0)
                        {
                            Log($"[周期训练] 完成：训练 {trainedAgents} 个智能体，使用 {totalSamples} 条样本，平均损失 {totalLoss / trainedAgents:F4}");
                        }
                        else
                        {
                            Log($"[周期训练] 跳过：没有可训练批次。忙碌 {skippedBusyAgents}，未初始化 {skippedUninitializedAgents}，空批次 {skippedNoDataAgents}");
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                }
                catch (Exception ex)
                {
                    Log($"[周期训练异常] {ex.Message}");
                    Log($"[周期训练异常-堆栈] {ex}");
                    OnError?.Invoke($"[周期训练异常] {ex.Message}");
                }
                finally
                {
                    for (int i = heldLocks.Count - 1; i >= 0; i--)
                    {
                        heldLocks[i].Release();
                    }

                    if (maintenanceLockHeld)
                    {
                        _maintenanceLock.Release();
                    }
                }
            });
        }

        public static float CalculateMaterialScore(Board board, bool isRed) => isRed ? board.RedMaterial : board.BlackMaterial;
        public static float GetBoardAdvantage(Board board)
        {
            float diff = board.RedMaterial - board.BlackMaterial;
            // 【BUG 10 优化】：降低阈值至 0.5 (约半个兵)，提高 PGN 数据标注灵敏度
            return diff > 0.5f ? 1.0f : (diff < -0.5f ? -1.0f : 0.0f);
        }

        public async Task ProcessDatasetAsync(string filePath)
        {
            if (IsTraining)
                return;
            if (_currentTrainingTask != null && !_currentTrainingTask.IsCompleted)
            {
                try
                {
                    await _currentTrainingTask;
                }
                catch { }
            }
            _isTraining = true;
            _cts = new CancellationTokenSource();
            _currentTrainingTask = Task.Run(async () =>
            {
                try
                {
                    string ext = Path.GetExtension(filePath).ToLower();
                    if (ext == ".csv")
                        ProcessCsvDataset(filePath, _cts.Token);
                    else if (ext == ".pgn" || ext == ".txt")
                        ProcessPgnDatasetStreaming(filePath, _cts.Token);
                }
                catch (Exception ex) { OnError?.Invoke($"[解析错误] {ex.Message}"); }
                finally
                {
                    if (_backgroundLoadTask != null && !_backgroundLoadTask.IsCompleted)
                    {
                        try
                        {
                            await _backgroundLoadTask;
                        }
                        catch { }
                    }
                    _isTraining = false;
                    OnTrainingStopped?.Invoke();
                }
            });
        }

        private void ProcessPgnDatasetStreaming(string filePath, CancellationToken token)
        {
            Log("[PGN 吞噬者] 正在以流式方式解析文件...");
            var rules = new ChineseChessRuleEngine();
            int totalProcessedGames = 0;

            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                StringBuilder blockBuilder = new StringBuilder();
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (token.IsCancellationRequested)
                        break;
                    if (line.StartsWith("[Event ") && blockBuilder.Length > 0)
                    {
                        ParseSinglePgnBlock(blockBuilder.ToString(), rules, ref totalProcessedGames);
                        blockBuilder.Clear();
                    }
                    blockBuilder.AppendLine(line);
                }
                if (!token.IsCancellationRequested && blockBuilder.Length > 0)
                    ParseSinglePgnBlock(blockBuilder.ToString(), rules, ref totalProcessedGames);
            }
            Log($"[PGN 吞噬者] 解析完毕！总吞噬 {totalProcessedGames} 局。");
        }

        private void ProcessCsvDataset(string filePath, CancellationToken token)
        {
            Log("[CSV 解析] 开始读取文件...");
            var rules = new ChineseChessRuleEngine();
            int totalGames = 0;
            string currentGameId = null;
            var redMoves = new List<(int turn, string move)>();
            var blackMoves = new List<(int turn, string move)>();

            using (var reader = new StreamReader(filePath, Encoding.UTF8))
            {
                reader.ReadLine();
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (token.IsCancellationRequested)
                        break;
                    var parts = line.Split(',');
                    if (parts.Length < 4)
                        continue;
                    string gameId = parts[0].Trim();
                    if (!int.TryParse(parts[1].Trim(), out int turn))
                        continue;
                    string side = parts[2].Trim().ToLower();
                    string move = parts[3].Trim();

                    if (currentGameId != null && gameId != currentGameId)
                    {
                        ProcessCsvGame(redMoves, blackMoves, rules, ref totalGames);
                        redMoves.Clear();
                        blackMoves.Clear();
                    }
                    currentGameId = gameId;
                    if (side == "red")
                        redMoves.Add((turn, move));
                    else
                        blackMoves.Add((turn, move));
                }

                if (!token.IsCancellationRequested && currentGameId != null && (redMoves.Count > 0 || blackMoves.Count > 0))
                {
                    ProcessCsvGame(redMoves, blackMoves, rules, ref totalGames);
                    redMoves.Clear();
                    blackMoves.Clear();
                }
            }
            Log($"[CSV 解析] 完成！总解析 {totalGames} 局。");
        }

        private void ParseSinglePgnBlock(string block, ChineseChessRuleEngine rules, ref int totalGames)
        {
            string reconstructedBlock = block.Trim();
            if (!reconstructedBlock.StartsWith("[Event "))
                reconstructedBlock = "[Event " + reconstructedBlock;

            float resultValue = 0.0f;
            bool hasExplicitResult = false;
            var resultMatch = System.Text.RegularExpressions.Regex.Match(reconstructedBlock, @"\[Result\s+""(.*?)""\]");
            if (resultMatch.Success)
            {
                string resStr = resultMatch.Groups[1].Value;
                if (resStr == "1-0")
                {
                    resultValue = 1.0f;
                    hasExplicitResult = true;
                }
                else if (resStr == "0-1")
                {
                    resultValue = -1.0f;
                    hasExplicitResult = true;
                }
                else if (resStr == "1/2-1/2")
                {
                    resultValue = 0.0f;
                    hasExplicitResult = true;
                }
            }

            string moveText = System.Text.RegularExpressions.Regex.Replace(reconstructedBlock, @"\[[^\]]*\]", "");
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"\{[^}]*\}", "");
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"\b\d+\.", "");
            // 去掉标准 PGN movetext 末尾的结果标记，防止被误判为非法着法导致 isComplete=false
            moveText = System.Text.RegularExpressions.Regex.Replace(moveText, @"(1-0|0-1|1/2-1/2|\*)\s*$", "");
            var moveStrings = moveText.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);

            var session = new GameRuleSession(rules);
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();
            bool isComplete = true;

            foreach (var rawMove in moveStrings)
            {
                if (ProcessSingleNotationMove(session, rawMove, gameHistory, out string normalizedUcci))
                {
                    standardizedMoves.Add(normalizedUcci);
                }
                else
                {
                    isComplete = false;
                    break;
                }
            }

            // 【数据质量修复】：如果对局截断，则不再信任整局结果，改用当前的材料差估分
            if (!isComplete)
            {
                resultValue = GetBoardAdvantage(session.Board);
            }
            else if (!hasExplicitResult)
            {
                resultValue = GetBoardAdvantage(session.Board);
            }

            if (isComplete && gameHistory.Count > 10)
            {
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                var masterData = new MasterGameData(examples, standardizedMoves);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string guid = Guid.NewGuid().ToString("N");
                File.WriteAllText(
                    Path.Combine(MasterBuffer.DataDir, $"pgn_game_{timestamp}_{guid}.json"),
                    JsonSerializer.Serialize(masterData),
                    Encoding.UTF8);

                totalGames++;
            }
        }

        private void ProcessCsvGame(List<(int turn, string move)> redMoves, List<(int turn, string move)> blackMoves, ChineseChessRuleEngine rules, ref int totalGames)
        {
            redMoves.Sort((a, b) => a.turn.CompareTo(b.turn));
            blackMoves.Sort((a, b) => a.turn.CompareTo(b.turn));
            var rawOrderedMoves = new List<string>();
            int maxTurn = Math.Max(redMoves.Count, blackMoves.Count);
            for (int i = 0; i < maxTurn; i++)
            {
                if (i < redMoves.Count)
                    rawOrderedMoves.Add(redMoves[i].move);
                if (i < blackMoves.Count)
                    rawOrderedMoves.Add(blackMoves[i].move);
            }

            var session = new GameRuleSession(rules);
            var gameHistory = new List<(float[] state, float[] policy, bool isRedTurn)>();
            var standardizedMoves = new List<string>();
            bool isComplete = true;
            foreach (var rawMove in rawOrderedMoves)
            {
                if (ProcessSingleNotationMove(session, rawMove, gameHistory, out string normalizedUcci))
                {
                    standardizedMoves.Add(normalizedUcci);
                }
                else
                {
                    isComplete = false;
                    break;
                }
            }
            if (isComplete && gameHistory.Count > 10)
            {
                float resultValue = GetBoardAdvantage(session.Board);
                // 注意：对于 CSV，我们本身就没有 Header Result，所以始终使用 GetBoardAdvantage 是安全的。
                // 但为了逻辑一致性，我们在这里显式体现。
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string guid = Guid.NewGuid().ToString("N");
                File.WriteAllText(
                    Path.Combine(MasterBuffer.DataDir, $"csv_game_{timestamp}_{guid}.json"),
                    JsonSerializer.Serialize(new MasterGameData(examples, standardizedMoves)),
                    Encoding.UTF8);
                totalGames++;
            }
        }

        private bool ProcessSingleNotationMove(GameRuleSession session, string rawMove, List<(float[] state, float[] policy, bool isRedTurn)> gameHistory, out string normalizedUcci)
        {
            normalizedUcci = string.Empty;
            if (!session.TryResolveNotation(rawMove, out var parsedMove, out normalizedUcci, out _))
                return false;

            var board = session.Board;
            var legalMoves = session.GetLegalMoves();

            bool isRed = board.IsRedTurn;
            float[] stateData;
            using (var stateTensor = StateEncoder.Encode(board))
            using (var state3D = stateTensor.squeeze(0))
            using (var stateCpu = state3D.cpu())
            {
                stateData = stateCpu.data<float>().ToArray();
            }

            float[] piData = new float[8100];
            int netIdx = parsedMove.ToNetworkIndex();
            float epsilon = 0.05f;
            float backgroundProb = epsilon / legalMoves.Count;
            foreach (var m in legalMoves)
            {
                int idx = m.ToNetworkIndex();
                if (idx >= 0 && idx < 8100)
                    piData[idx] = backgroundProb;
            }
            if (netIdx >= 0 && netIdx < 8100)
                piData[netIdx] = (1.0f - epsilon) + backgroundProb;

            gameHistory.Add((stateData, isRed ? piData : StateEncoder.FlipPolicy(piData), isRed));
            session.ApplyMove(parsedMove, normalizedUcci);
            return true;
        }

        private void Log(string msg)
        {
            Volatile.Write(ref _lastLogUtcTicks, DateTimeOffset.UtcNow.UtcTicks);

            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(_runtimeLogPath)!);
                string line = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} {msg}{Environment.NewLine}";
                lock (_runtimeLogLock)
                {
                    File.AppendAllText(_runtimeLogPath, line, Encoding.UTF8);
                }
            }
            catch
            {
            }

            OnLog?.Invoke(msg);
        }
    }
}
