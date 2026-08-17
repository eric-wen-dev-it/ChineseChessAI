using ChineseChessAI.Core;
using ChineseChessAI.MCTS;
using ChineseChessAI.NeuralNetwork;
using ChineseChessAI.Traditional;
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
        private const int LoadedAgentCacheLimit = 12;
        private readonly object _inFlightGamesLock = new object();
        private static readonly TimeSpan LeagueGameTimeout = TimeSpan.FromMinutes(30);
        private static readonly TimeSpan LeagueDrainWaitTimeout = LeagueGameTimeout + TimeSpan.FromMinutes(3);
        private static readonly TimeSpan DrainProgressLogInterval = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan WatchdogCheckInterval = TimeSpan.FromMinutes(10);
        private static readonly TimeSpan WatchdogStaleLogThreshold = TimeSpan.FromMinutes(45);
        private static readonly TimeSpan LeagueShutdownWaitTimeout = TimeSpan.FromMinutes(1);
        private static readonly string _leagueTimeoutRecordsDir = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "league_timeout_records");
        private static readonly string _leaguePendingDataDir = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "league_pending");
        private static readonly string _masterTeacherDataDir = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "master_teacher_data");
        private static readonly string _masterTeacherBadDir = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "master_teacher_bad");
        private const int MasterReplayBufferCapacity = 4500000;   // ~5.5KB/样本 ≈ 24G 内存预算;每次种群重组后全池随机换血重装
        private const int LeagueReplayBufferCapacity = 150000;
        private const int MaxScoredMasterLoadFiles = 10000;
        private const int MaxRawMasterLoadFiles = 25000;
        private const int MaxHistoricalLeagueLoadFiles = 1500;
        private const int MaxLeagueTrainingGames = 5000;
        private const int TeacherBackfillNodes = 10000;
        private const int MasterTeacherBackfillNodes = 10000;
        // 大师谱打分的样本级并发度,须小于 PikafishAdjudicator.MaxClientCount(12),
        // 留余量给对局裁决和联赛谱打分抢槽。
        private const int MasterTeacherBackfillConcurrency = 8;
        private static readonly TimeSpan TeacherBackfillIdleDelay = TimeSpan.FromSeconds(5);
        private static readonly TimeSpan TeacherBackfillFileAge = TimeSpan.FromSeconds(3);
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
        private Task? _teacherBackfillTask;
        private Task? _masterTeacherBackfillTask;
        private long _lastLogUtcTicks = DateTimeOffset.UtcNow.UtcTicks;
        private int _watchdogRestartRequested;
        private volatile bool _skipAgentDisposeOnNextStart;

        private sealed record LeagueRunOptions(
            int PopulationSize,
            int MaxMoves,
            int ExploreMoves,
            float MaterialBias,
            int TraditionalAgentCount,
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
                string currentLogPath = RuntimeDiagnostics.CurrentLogPath;
                if (File.Exists(currentLogPath))
                {
                    DateTime fileWriteTimeUtc = File.GetLastWriteTimeUtc(currentLogPath);
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
                options.TraditionalAgentCount,
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
            ModelPaths.EnsureBestModelsDirectory(AppDomain.CurrentDomain.BaseDirectory);
            MasterBuffer = new ReplayBuffer(MasterReplayBufferCapacity, "data/master_data");
            LeagueBuffer = new ReplayBuffer(LeagueReplayBufferCapacity, "data/league_data");

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
            int traditionalAgentCount = 0,
            int populationRefreshInterval = 0,
            int? maxPopulationRefreshCycles = null)
        {
            if (populationSize > 100)
                throw new ArgumentException("出于内存限制与并发安全考量，联赛人口数量不能超过 100。", nameof(populationSize));
            if (populationSize < 2)
                throw new ArgumentException("联赛人口数量必须大于等于 2。", nameof(populationSize));
            traditionalAgentCount = Math.Clamp(traditionalAgentCount, 0, populationSize - 1);

            if (populationRefreshInterval <= 0)
                populationRefreshInterval = populationSize * 100;   // ×4(原 25):每人每轮 ~200 盘,Elo 噪声 ±57→±28,淘汰决策才有信号(50 盘时噪声大于梯队真实差距 15-30)
            populationRefreshInterval = Math.Max(populationRefreshInterval, populationSize * 4);
            if (maxPopulationRefreshCycles.HasValue && maxPopulationRefreshCycles.Value <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxPopulationRefreshCycles));
            var runOptions = new LeagueRunOptions(
                populationSize,
                maxMoves,
                exploreMoves,
                materialBias,
                traditionalAgentCount,
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

            var leagueManager = new LeagueManager(populationSize, traditionalAgentCount);
            _leagueManager = leagueManager;
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
                    Log($"=== 万王之王：{populationSize} 智能体联赛启动（传统搜索 {traditionalAgentCount} 个）===");
                    Directory.CreateDirectory(_leaguePendingDataDir);
                    Directory.CreateDirectory(LeagueBuffer.DataDir);
                    Directory.CreateDirectory(_masterTeacherDataDir);
                    Directory.CreateDirectory(_masterTeacherBadDir);

                    _teacherBackfillTask = Task.Run(
                        () => RunTeacherBackfillLoopAsync(runCts.Token),
                        runCts.Token);
                    _masterTeacherBackfillTask = Task.Run(
                        () => RunMasterTeacherBackfillLoopAsync(runCts.Token),
                        runCts.Token);

                    // 将数据装载放入独立的后台任务，不阻塞联赛和对局的立即启动
                    _backgroundLoadTask = Task.Run(async () =>
                    {
                        try
                        {
                            Log("[后台任务] 正在静默装载大师数据与历史联赛数据...");

                            Log($"[MemoryBudget] master_capacity={MasterReplayBufferCapacity}, league_capacity={LeagueReplayBufferCapacity}, scored_master_files={MaxScoredMasterLoadFiles}, raw_master_files={MaxRawMasterLoadFiles}, league_files={MaxHistoricalLeagueLoadFiles}.");
                            var (masterGames, masterSamples) = await ReloadMasterBufferAsync(startTime, runCts.Token);
                            var (leagueSamples, leagueGames) = await LeagueBuffer.LoadOldSamplesAsync(MaxHistoricalLeagueLoadFiles, randomize: true, logAction: Log, onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r), cancellationToken: runCts.Token, cutoffTime: startTime);

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
                            await PerformDiverseTrainingAsync(leagueManager, runCts.Token);
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
                            await PerformPopulationRefreshAsync(leagueManager, runCts.Token);
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

                            // 种群重组后大师池全量换血:后台随机重抽一池新谱,不阻塞对局
                            if (_backgroundLoadTask == null || _backgroundLoadTask.IsCompleted)
                            {
                                _backgroundLoadTask = Task.Run(async () =>
                                {
                                    try
                                    {
                                        Log("[后台任务] 种群重组完成,大师数据换血:重新随机抽样装载...");
                                        var (games, samples) = await ReloadMasterBufferAsync(DateTime.Now, runCts.Token);
                                        Log($"[后台装载完成] 大师数据换血: {games} 局 ({samples} 条)");
                                    }
                                    catch (Exception ex) { Log($"[后台装载异常] {ex.Message}"); }
                                }, runCts.Token);
                            }
                            else
                            {
                                Log("[后台任务] 上一轮装载尚未完成,跳过本次大师数据换血。");
                            }

                            continue;
                        }

                        var reservedAgentIds = GetReservedAgentIdsSnapshot();
                        if (!leagueManager.TryPickMatch(reservedAgentIds, out var agentMetaA, out var agentMetaB))
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
                                        bool countCompletedGameForSchedule = false;
                                        try
                                        {
                                            using var engineA = CreateLeagueGameEngine(agentMetaA);
                                            using var engineB = CreateLeagueGameEngine(agentMetaB);

                                            var selfPlay = new SelfPlay(engineA.Engine, engineB.Engine, currentMaxMoves, exploreMoves, materialBias,
                                                                        lowTempA: agentMetaA.Temperature, lowTempB: agentMetaB.Temperature,
                                                                        simsA: GetSearchBudget(agentMetaA), simsB: GetSearchBudget(agentMetaB));

                                            Log($"[对局 #{currentId} 开始] Agent_{agentMetaA.Id}(ELO:{agentMetaA.Elo:F0} {FormatEngineDna(agentMetaA)}) " +
                                                $"VS Agent_{agentMetaB.Id}(ELO:{agentMetaB.Elo:F0} {FormatEngineDna(agentMetaB)})");

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
                                                countCompletedGameForSchedule = true;
                                                float resA = aIsRed ? result.RatingResultForRed : -result.RatingResultForRed;

                                                // 【修复 P1】：捕获赛前 ELO 以确保更新公平
                                                double eloABefore = agentMetaA.Elo;
                                                double eloBBefore = agentMetaB.Elo;

                                                lock (leagueManager)
                                                {
                                                    leagueManager.UpdateResult(agentMetaA.Id, resA, eloBBefore);
                                                    leagueManager.UpdateResult(agentMetaB.Id, -resA, eloABefore);
                                                }

                                                var combinedExamples = new List<TrainingExample>(result.ExamplesA.Count + result.ExamplesB.Count);
                                                combinedExamples.AddRange(result.ExamplesA);
                                                combinedExamples.AddRange(result.ExamplesB);

                                                if (combinedExamples.Count > 0)
                                                {
                                                    var moveHistoryUcci = result.MoveHistory.Select(NotationConverter.MoveToUcci).ToList();
                                                    SavePendingLeagueGame(new MasterGameData(combinedExamples, moveHistoryUcci)
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
                                            if (countCompletedGameForSchedule)
                                                Interlocked.Increment(ref completedGameCounter);
                                            lockSecond.Release();
                                        }
                                    }
                                    finally { lockFirst.Release(); }
                                }
                                catch (OperationCanceledException) when (runCts.IsCancellationRequested)
                                {
                                    Log($"[对局 #{currentId} 取消] 联赛停止或重启，取消等待/搜索。");
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
                            leagueManager.SaveMetadata();
                            var top = leagueManager.GetTopNeuralAgents(5);
                            Log("--- [当前排名 Top 5 - Neural only] ---");
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
                    if (_teacherBackfillTask != null && !_teacherBackfillTask.IsCompleted)
                    {
                        try
                        {
                            await _teacherBackfillTask.WaitAsync(TimeSpan.FromSeconds(5));
                        }
                        catch { }
                    }
                    if (_masterTeacherBackfillTask != null && !_masterTeacherBackfillTask.IsCompleted)
                    {
                        try
                        {
                            await _masterTeacherBackfillTask.WaitAsync(TimeSpan.FromSeconds(5));
                        }
                        catch { }
                    }
                    _isTraining = false;
                    leagueManager.SaveMetadata();
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
                bool modelLoaded = false;
                lock (GetFileLock(meta.ModelPath))
                {
                    if (File.Exists(meta.ModelPath))
                    {
                        try
                        {
                            pa.Model.load(meta.ModelPath);
                            modelLoaded = true;
                        }
                        catch (Exception ex) when (ex is not OperationCanceledException)
                        {
                            string quarantinedPath = QuarantineCorruptModelFile(meta.ModelPath);
                            Log($"[模型损坏] Agent_{meta.Id} 模型文件无法加载: {meta.ModelPath}");
                            Log($"[模型损坏] 已隔离到: {quarantinedPath}");
                            Log($"[模型损坏-堆栈] {ex}");
                            // 模型损坏后，旧优化器状态对应的是失效的参数，必须一起隔离。
                            QuarantineOptimizerStateFiles(meta.ModelPath);
                        }
                    }
                }

                pa.CompleteInit(); // to(CUDA) + new Trainer(Model)，必须在 load() 之后

                if (modelLoaded)
                {
                    // 【优化器状态持久化】：模型权重和 Adam 动量必须配套使用，
                    // 否则 agent 每次重载会回到"高 LR + 无动量"的状态，长期训练高噪声。
                    string optimizerPath = GetOptimizerStatePath(meta.ModelPath);
                    string sidecarPath = GetOptimizerSidecarPath(meta.ModelPath);
                    lock (GetFileLock(meta.ModelPath))
                    {
                        if (pa.Trainer.TryLoadOptimizerState(optimizerPath, sidecarPath))
                        {
                            Log($"[优化器恢复] Agent_{meta.Id}: {Path.GetFileName(optimizerPath)}");
                        }
                    }
                }

                return pa;
            }, LazyThreadSafetyMode.ExecutionAndPublication)).Value;
            TouchAgent(meta.Id);
            return agent;
        }

        // 优化器状态文件与模型文件同目录共存，命名规则：foo.pt → foo.optim, foo.optim.json
        internal static string GetOptimizerStatePath(string modelPath) => modelPath + ".optim";
        internal static string GetOptimizerSidecarPath(string modelPath) => modelPath + ".optim.json";

        private void QuarantineOptimizerStateFiles(string modelPath)
        {
            foreach (string path in new[] { GetOptimizerStatePath(modelPath), GetOptimizerSidecarPath(modelPath) })
            {
                if (!File.Exists(path))
                    continue;
                try
                {
                    string quarantined = path + $".corrupt_{DateTime.Now:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";
                    File.Move(path, quarantined, overwrite: false);
                    Log($"[优化器隔离] {Path.GetFileName(path)} → {Path.GetFileName(quarantined)}");
                }
                catch (Exception ex)
                {
                    Log($"[优化器隔离失败] {path}: {ex.Message}");
                    try
                    {
                        File.Delete(path);
                    }
                    catch
                    {
                    }
                }
            }
        }

        private void SaveAgentModelAndOptimizer(PersistentAgent pa, AgentMetadata meta)
        {
            lock (GetFileLock(meta.ModelPath))
            {
                ModelManager.SaveModel(pa.Model, meta.ModelPath);
                pa.Trainer.SaveOptimizerState(GetOptimizerStatePath(meta.ModelPath), GetOptimizerSidecarPath(meta.ModelPath));
            }
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

        private async Task PerformPopulationRefreshAsync(LeagueManager leagueManager, CancellationToken token)
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

                foreach (int agentId in leagueManager.GetAllAgentIds())
                {
                    var agentLock = GetAgentActiveLock(agentId);
                    await agentLock.WaitAsync(token);
                    heldLocks.Add(agentLock);
                }

                lock (_gpuTrainingLock)
                {
                    FlushLoadedModelsToDisk(leagueManager);

                    int populationSize = leagueManager.GetPopulationSize();
                    int eliteCount = Math.Clamp(populationSize / 10, 1, Math.Max(1, populationSize - 3));
                    // 竞争者 3/10→4/10、多样性 1/5→1/4:淘汰率 ~35%→~20%。测量噪声下重淘汰≈随机漂变,轻淘汰+长周期让选择有信号
                    int contenderKeepCount = Math.Clamp(populationSize * 4 / 10, 1, Math.Max(1, populationSize - eliteCount - 2));
                    int diverseKeepCount = Math.Clamp(populationSize / 4, 1, Math.Max(1, populationSize - eliteCount - contenderKeepCount - 1));
                    int parentPoolSize = Math.Clamp(Math.Min(10, Math.Max(4, populationSize / 5)), 1, populationSize);

                    int replacementCount = Math.Max(0, populationSize - eliteCount - contenderKeepCount - diverseKeepCount);
                    int immigrantCount = replacementCount > 0 ? Math.Clamp(Math.Max(1, replacementCount / 5), 1, replacementCount) : 0;

                    var refresh = leagueManager.RefreshPopulation(
                        eliteCount,
                        contenderKeepCount,
                        diverseKeepCount,
                        parentPoolSize,
                        immigrantCount,
                        newbornProtectionGames: 100);

                    RefreshAgentPool(refresh.ReplacedAgentIds);
                    TrimIdleAgentPool();

                    if (refresh.Replaced > 0)
                    {
                        Log($"[种群重组] 完成：精英保留 {refresh.EliteKept}，竞争者保留 {refresh.ContenderKept}，多样性保留 {refresh.DiverseKept}，新生保护 {refresh.NewbornProtected}，重建 {refresh.Replaced}（后代 {refresh.OffspringCreated}，移民 {refresh.ImmigrantsCreated}）。");
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

        private void FlushLoadedModelsToDisk(LeagueManager leagueManager)
        {
            foreach (var agentEntry in _agentPool)
            {
                if (!agentEntry.Value.IsValueCreated)
                {
                    continue;
                }

                var meta = leagueManager.GetAgentMeta(agentEntry.Key);
                if (meta == null)
                {
                    continue;
                }

                SaveAgentModelAndOptimizer(agentEntry.Value.Value, meta);
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
                    ReplayBuffer.WriteTextAtomic(filePath, JsonSerializer.Serialize(record, new JsonSerializerOptions
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

        // 2026-08-24 标尺切换增强档(用户拍板):Elo 重定价窗口自此重开,曲线斜率须从本部署点重起。
        private int _traditionalBenchmarkUpgradeLogged;

        private LeagueEngineHandle CreateLeagueGameEngine(AgentMetadata meta)
        {
            if (IsTraditionalAgent(meta))
            {
                var book = OpeningBook.LoadDefaultCache(maxPly: 24);
                var options = new TraditionalEngineOptions
                {
                    OpeningBook = book,
                    OpeningBookMode = book.PositionCount > 0 ? OpeningBookMode.Weighted : OpeningBookMode.Off,
                    MoveOrderingBook = OpeningBook.LoadDefaultCache(maxPly: 80, fileName: "master_move_ordering.json"),
                    MasterKnowledgeBook = MasterKnowledgeBook.LoadDefaultCache(maxPly: 120),
                    SkipPerpetualCheckAtRoot = true,
                    MateSearchPly = 3
                }.WithEnhancedQuiescence();
                if (Interlocked.Exchange(ref _traditionalBenchmarkUpgradeLogged, 1) == 0)
                    Log("[标尺] 传统引擎增强档已启用(静搜TT/delta/SEE/将军限深2, qdepth=12)。");
                return new LeagueEngineHandle(new TraditionalGameEngineAdapter(new TraditionalEngine(options), quiescenceDepth: 12));
            }

            var persistentAgent = GetOrAddAgent(meta);
            var mcts = new MCTSEngine(persistentAgent.Model, batchSize: 16, cPuct: meta.Cpuct);
            return new LeagueEngineHandle(new MctsGameEngineAdapter(mcts));
        }

        private static bool IsTraditionalAgent(AgentMetadata meta)
        {
            return string.Equals(meta.EngineKind, "Traditional", StringComparison.OrdinalIgnoreCase);
        }

        private static int GetSearchBudget(AgentMetadata meta)
        {
            return IsTraditionalAgent(meta)
                ? Math.Clamp(meta.TraditionalDepth, 1, 12)
                : meta.MctsSimulations;
        }

        private static string FormatEngineDna(AgentMetadata meta)
        {
            return IsTraditionalAgent(meta)
                ? $"Traditional:D{Math.Clamp(meta.TraditionalDepth, 1, 12)}"
                : $"DNA:S{meta.MctsSimulations}/C{meta.Cpuct:F1}/T{meta.Temperature:F1}";
        }

        private sealed class LeagueEngineHandle : IDisposable
        {
            public IGameEngine Engine { get; }

            public LeagueEngineHandle(IGameEngine engine)
            {
                Engine = engine;
            }

            public void Dispose()
            {
                if (Engine is IDisposable disposable)
                    disposable.Dispose();
            }
        }

        private void SavePendingLeagueGame(MasterGameData gameData)
        {
            try
            {
                Directory.CreateDirectory(_leaguePendingDataDir);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string filePath = Path.Combine(_leaguePendingDataDir, $"game_{timestamp}_{Guid.NewGuid():N}.json");
                ReplayBuffer.WriteTextAtomic(filePath, JsonSerializer.Serialize(gameData));
                Log($"[TeacherBackfill] 已入队待打分对局: {Path.GetFileName(filePath)} samples={gameData.Examples.Count}");
            }
            catch (Exception ex)
            {
                Log($"[TeacherBackfill] 待打分对局保存失败: {ex.Message}");
                OnError?.Invoke($"[TeacherBackfill] 待打分对局保存失败: {ex.Message}");
            }
        }

        private async Task RunTeacherBackfillLoopAsync(CancellationToken token)
        {
            Directory.CreateDirectory(_leaguePendingDataDir);
            Directory.CreateDirectory(LeagueBuffer.DataDir);
            Log($"[TeacherBackfill] 后台全量打分启动: pending={_leaguePendingDataDir}, final={LeagueBuffer.DataDir}");

            while (!token.IsCancellationRequested)
            {
                try
                {
                    var files = Directory.GetFiles(_leaguePendingDataDir, "*.json")
                        .Select(path => new FileInfo(path))
                        .OrderBy(info => info.CreationTimeUtc)
                        .ThenBy(info => info.Name, StringComparer.Ordinal)
                        .ToList();

                    bool processedAny = false;
                    foreach (var file in files)
                    {
                        token.ThrowIfCancellationRequested();
                        if (DateTime.UtcNow - file.LastWriteTimeUtc < TeacherBackfillFileAge)
                            continue;

                        processedAny = true;
                        bool success = await TryBackfillTeacherFileAsync(file.FullName, token).ConfigureAwait(false);
                        if (!success)
                            await Task.Delay(TeacherBackfillIdleDelay, token).ConfigureAwait(false);
                    }

                    if (!processedAny)
                        await Task.Delay(TeacherBackfillIdleDelay, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Log($"[TeacherBackfill] 后台打分异常: {ex.Message}");
                    await Task.Delay(TeacherBackfillIdleDelay, token).ConfigureAwait(false);
                }
            }

            Log("[TeacherBackfill] 后台全量打分停止。");
        }

        private async Task<bool> TryBackfillTeacherFileAsync(string pendingPath, CancellationToken token)
        {
            string fileName = Path.GetFileName(pendingPath);
            string json;
            try
            {
                json = File.ReadAllText(pendingPath);
            }
            catch (Exception ex)
            {
                Log($"[TeacherBackfill] 读取失败 {fileName}: {ex.Message}");
                return false;
            }

            MasterGameData? gameData;
            try
            {
                gameData = JsonSerializer.Deserialize<MasterGameData>(json);
            }
            catch (Exception ex)
            {
                Log($"[TeacherBackfill] JSON 解析失败 {fileName}: {ex.Message}");
                return MovePendingFileAside(pendingPath, ".invalid");
            }

            if (gameData?.Examples == null || gameData.Examples.Count == 0)
            {
                Log($"[TeacherBackfill] 空对局，跳过 {fileName}");
                return MovePendingFileAside(pendingPath, ".invalid");
            }

            var scoredExamples = new List<TrainingExample>(gameData.Examples.Count);
            int valueLabels = 0;
            int policyLabels = 0;
            int skippedLabels = 0;
            Log($"[TeacherBackfill] 开始全量打分 {fileName}: samples={gameData.Examples.Count}");

            for (int i = 0; i < gameData.Examples.Count; i++)
            {
                token.ThrowIfCancellationRequested();
                TrainingExample example = gameData.Examples[i];
                if (example.UcciHistoryBefore == null)
                {
                    Log($"[TeacherBackfill] 缺少 UcciHistoryBefore，降级保存无教师标签样本 {fileName} sample={i}");
                    skippedLabels++;
                    scoredExamples.Add(StripTeacherHistory(example));
                    continue;
                }

                bool redToMove = example.UcciHistoryBefore.Length % 2 == 0;
                PikafishTeacherAnalysis? analysis = await PikafishAdjudicator.TryAnalyzeAsync(
                    example.UcciHistoryBefore,
                    redToMove,
                    TeacherBackfillNodes,
                    token).ConfigureAwait(false);

                if (analysis == null)
                {
                    Log($"[TeacherBackfill] Pikafish 未返回结果，降级保存无教师标签样本 {fileName} sample={i}");
                    skippedLabels++;
                    scoredExamples.Add(StripTeacherHistory(example));
                    continue;
                }

                ActionProb[]? teacherPolicy = BuildTeacherSparsePolicy(analysis.BestMove, redToMove);
                if (teacherPolicy != null)
                    policyLabels++;

                valueLabels++;
                scoredExamples.Add(example with
                {
                    TeacherValue = analysis.ValueForCurrentPlayer,
                    TeacherSparsePolicy = teacherPolicy,
                    UcciHistoryBefore = null
                });
            }

            var scoredGame = gameData with
            {
                Examples = scoredExamples
            };

            string finalPath = Path.Combine(LeagueBuffer.DataDir, fileName);
            try
            {
                ReplayBuffer.WriteTextAtomic(finalPath, JsonSerializer.Serialize(scoredGame));
                File.Delete(pendingPath);
                LeagueBuffer.AddGame(scoredGame, saveToDisk: false);
                Log($"[TeacherBackfill] 完成全量打分/降级入库 {fileName}: values={valueLabels}, policies={policyLabels}, skipped={skippedLabels}");
                return true;
            }
            catch (Exception ex)
            {
                Log($"[TeacherBackfill] 写入完成目录失败 {fileName}: {ex.Message}");
                return false;
            }
        }

        private static ActionProb[]? BuildTeacherSparsePolicy(string bestMove, bool redToMove)
        {
            if (string.IsNullOrWhiteSpace(bestMove)
                || bestMove == "0000"
                || bestMove.Equals("(none)", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            Move? move = NotationConverter.UcciToMove(bestMove);
            if (!move.HasValue)
                return null;

            int idx = move.Value.ToNetworkIndex();
            if (idx < 0 || idx >= 8100)
                return null;

            if (redToMove)
                return new[] { new ActionProb(idx, 1.0f) };

            float[] policy = new float[8100];
            policy[idx] = 1.0f;
            return StateEncoder.FlipPolicy(policy)
                .Select((p, i) => new ActionProb(i, p))
                .Where(x => x.Prob > 0)
                .ToArray();
        }

        private static TrainingExample StripTeacherHistory(TrainingExample example)
        {
            return example with { UcciHistoryBefore = null };
        }

        private bool MovePendingFileAside(string pendingPath, string suffix)
        {
            try
            {
                string target = pendingPath + suffix;
                File.Move(pendingPath, target, overwrite: true);
                return true;
            }
            catch (Exception ex)
            {
                Log($"[TeacherBackfill] 隔离待打分文件失败 {Path.GetFileName(pendingPath)}: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 清空大师池并从磁盘全目录随机重新抽样装载（教师打分谱优先，生谱排重补足）。
        /// 启动时与每次种群重组后各调用一次，使全量语料随重组周期轮转进入训练。
        /// </summary>
        private async Task<(int games, int samples)> ReloadMasterBufferAsync(DateTime cutoffTime, CancellationToken token)
        {
            var scoredMasterNames = Directory.Exists(_masterTeacherDataDir)
                ? Directory.GetFiles(_masterTeacherDataDir, "*.json")
                    .Select(Path.GetFileName)
                    .Where(name => !string.IsNullOrEmpty(name))
                    .ToHashSet(StringComparer.OrdinalIgnoreCase)!
                : new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            MasterBuffer.Clear();
            var (scoredMasterSamples, scoredMasterGames) = await MasterBuffer.LoadOldSamplesAsync(MaxScoredMasterLoadFiles, randomize: true, logAction: Log, onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r), cancellationToken: token, cutoffTime: cutoffTime, sourceDataDir: _masterTeacherDataDir);
            var (rawMasterSamples, rawMasterGames) = await MasterBuffer.LoadOldSamplesAsync(MaxRawMasterLoadFiles, randomize: true, logAction: Log, onAuditFailure: (h, m, r) => OnAuditFailureRequested?.Invoke(h, m, r), cancellationToken: token, cutoffTime: cutoffTime, excludedFileNames: scoredMasterNames);
            return (scoredMasterGames + rawMasterGames, scoredMasterSamples + rawMasterSamples);
        }

        private async Task RunMasterTeacherBackfillLoopAsync(CancellationToken token)
        {
            Directory.CreateDirectory(MasterBuffer.DataDir);
            Directory.CreateDirectory(_masterTeacherDataDir);
            Directory.CreateDirectory(_masterTeacherBadDir);
            Log($"[MasterTeacherBackfill] 后台大师谱全量打分启动: source={MasterBuffer.DataDir}, final={_masterTeacherDataDir}");

            // 待打分队列按批补充:全目录(65万+文件)一次扫描+两次 File.Exists/文件的成本
            // 从"每局一次"摊薄到"每 256 局一次"。TryBackfill 入口有幂等重查,过期条目无害。
            var pendingFiles = new Queue<string>();
            while (!token.IsCancellationRequested)
            {
                try
                {
                    if (pendingFiles.Count == 0)
                    {
                        foreach (string path in Directory.EnumerateFiles(MasterBuffer.DataDir, "*.json")
                            .Select(path => new FileInfo(path))
                            .Where(info =>
                                !File.Exists(Path.Combine(_masterTeacherDataDir, info.Name)) &&
                                !File.Exists(Path.Combine(_masterTeacherBadDir, info.Name + ".invalid")))
                            .OrderBy(info => info.CreationTimeUtc)
                            .ThenBy(info => info.Name, StringComparer.Ordinal)
                            .Take(256)
                            .Select(info => info.FullName))
                        {
                            pendingFiles.Enqueue(path);
                        }

                        if (pendingFiles.Count == 0)
                        {
                            await Task.Delay(TeacherBackfillIdleDelay, token).ConfigureAwait(false);
                            continue;
                        }
                    }

                    bool success = await TryBackfillMasterTeacherFileAsync(pendingFiles.Dequeue(), token).ConfigureAwait(false);
                    if (!success)
                        await Task.Delay(TeacherBackfillIdleDelay, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Log($"[MasterTeacherBackfill] 后台大师谱打分异常: {ex.Message}");
                    await Task.Delay(TeacherBackfillIdleDelay, token).ConfigureAwait(false);
                }
            }

            Log("[MasterTeacherBackfill] 后台大师谱全量打分停止。");
        }

        private async Task<bool> TryBackfillMasterTeacherFileAsync(string sourcePath, CancellationToken token)
        {
            string fileName = Path.GetFileName(sourcePath);
            string finalPath = Path.Combine(_masterTeacherDataDir, fileName);
            string badPath = Path.Combine(_masterTeacherBadDir, fileName + ".invalid");

            if (File.Exists(finalPath) || File.Exists(badPath))
                return true;

            MasterGameData? gameData;
            try
            {
                string json = File.ReadAllText(sourcePath);
                gameData = JsonSerializer.Deserialize<MasterGameData>(json);
            }
            catch (Exception ex)
            {
                Log($"[MasterTeacherBackfill] 读取/解析失败 {fileName}: {ex.Message}");
                return WriteMasterBadMarker(badPath, ex.Message);
            }

            if (gameData?.Examples == null || gameData.Examples.Count == 0)
            {
                Log($"[MasterTeacherBackfill] 空大师谱，跳过 {fileName}");
                return WriteMasterBadMarker(badPath, "empty examples");
            }

            if (gameData.MoveHistoryUcci == null || gameData.MoveHistoryUcci.Count == 0)
            {
                Log($"[MasterTeacherBackfill] 缺少 MoveHistoryUcci，无法打分 {fileName}");
                return WriteMasterBadMarker(badPath, "missing MoveHistoryUcci");
            }

            if (gameData.Examples.Count > gameData.MoveHistoryUcci.Count + 1)
            {
                Log($"[MasterTeacherBackfill] 样本数超过棋谱长度，无法可靠打分 {fileName}: samples={gameData.Examples.Count}, moves={gameData.MoveHistoryUcci.Count}");
                return WriteMasterBadMarker(badPath, "examples exceed move history");
            }

            var scoredResults = new TrainingExample[gameData.Examples.Count];
            int valueLabels = 0;
            int policyLabels = 0;
            Log($"[MasterTeacherBackfill] 开始大师谱全量打分 {fileName}: samples={gameData.Examples.Count}");

            await Parallel.ForEachAsync(
                Enumerable.Range(0, gameData.Examples.Count),
                new ParallelOptions { MaxDegreeOfParallelism = MasterTeacherBackfillConcurrency, CancellationToken = token },
                async (i, ct) =>
                {
                    TrainingExample example = gameData.Examples[i];
                    string[] historyBefore = example.UcciHistoryBefore
                        ?? gameData.MoveHistoryUcci.Take(i).ToArray();
                    bool redToMove = historyBefore.Length % 2 == 0;

                    PikafishTeacherAnalysis? analysis = await PikafishAdjudicator.TryAnalyzeAsync(
                        historyBefore,
                        redToMove,
                        MasterTeacherBackfillNodes,
                        ct).ConfigureAwait(false);

                    if (analysis == null)
                    {
                        Log($"[MasterTeacherBackfill] Pikafish 未返回结果，降级保存无教师标签样本 {fileName} sample={i}");
                        scoredResults[i] = StripTeacherHistory(example);
                        return;
                    }

                    ActionProb[]? teacherPolicy = BuildTeacherSparsePolicy(analysis.BestMove, redToMove);
                    if (teacherPolicy != null)
                        Interlocked.Increment(ref policyLabels);
                    Interlocked.Increment(ref valueLabels);

                    scoredResults[i] = example with
                    {
                        TeacherValue = analysis.ValueForCurrentPlayer,
                        TeacherSparsePolicy = teacherPolicy,
                        UcciHistoryBefore = null
                    };
                }).ConfigureAwait(false);

            var scoredGame = gameData with
            {
                Examples = scoredResults.ToList()
            };

            try
            {
                ReplayBuffer.WriteTextAtomic(finalPath, JsonSerializer.Serialize(scoredGame));
                Log($"[MasterTeacherBackfill] 完成大师谱全量打分 {fileName}: values={valueLabels}, policies={policyLabels}");
                return true;
            }
            catch (Exception ex)
            {
                Log($"[MasterTeacherBackfill] 写入完成目录失败 {fileName}: {ex.Message}");
                return false;
            }
        }

        private bool WriteMasterBadMarker(string badPath, string reason)
        {
            try
            {
                Directory.CreateDirectory(_masterTeacherBadDir);
                ReplayBuffer.WriteTextAtomic(badPath, reason);
                return true;
            }
            catch (Exception ex)
            {
                Log($"[MasterTeacherBackfill] 写入坏文件标记失败 {Path.GetFileName(badPath)}: {ex.Message}");
                return false;
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

        private async Task PerformDiverseTrainingAsync(LeagueManager leagueManager, CancellationToken token)
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

                    foreach (int agentId in leagueManager.GetAllAgentIds())
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

                    int populationSize = leagueManager.GetPopulationSize();
                    Log($"[周期训练] 开始：大师样本 {MasterBuffer.Count}，联赛样本 {LeagueBuffer.Count}，训练智能体 {populationSize}");

                    lock (_gpuTrainingLock)
                    {
                        const int batchSize = 256;
                        const int trainingEpochs = 2;
                        const float masterRatio = 0.3f;
                        const float leagueRatio = 0.7f;

                        foreach (int agentId in leagueManager.GetAllAgentIds())
                        {
                            if (token.IsCancellationRequested)
                                return;

                            var meta = leagueManager.GetAgentMeta(agentId);
                            if (meta == null)
                            {
                                skippedUninitializedAgents++;
                                continue;
                            }

                            if (IsTraditionalAgent(meta))
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
                                SaveAgentModelAndOptimizer(pa, meta);
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
                resultValue = BoardEvaluation.AdjudicateDrawByMaterial(session.Board);
            }
            else if (!hasExplicitResult)
            {
                resultValue = BoardEvaluation.AdjudicateDrawByMaterial(session.Board);
            }

            if (gameHistory.Count > 10)
            {
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                var masterData = new MasterGameData(examples, standardizedMoves);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string guid = Guid.NewGuid().ToString("N");
                ReplayBuffer.WriteTextAtomic(
                    Path.Combine(MasterBuffer.DataDir, $"pgn_game_{timestamp}_{guid}.json"),
                    JsonSerializer.Serialize(masterData));

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
            foreach (var rawMove in rawOrderedMoves)
            {
                if (ProcessSingleNotationMove(session, rawMove, gameHistory, out string normalizedUcci))
                {
                    standardizedMoves.Add(normalizedUcci);
                }
                else
                {
                    break;
                }
            }
            if (gameHistory.Count > 10)
            {
                float resultValue = BoardEvaluation.AdjudicateDrawByMaterial(session.Board);
                var examples = gameHistory.Select(step =>
                {
                    var sparse = step.policy.Select((p, i) => new ActionProb(i, p)).Where(x => x.Prob > 0).ToArray();
                    return new TrainingExample(step.state, sparse, step.isRedTurn ? resultValue : -resultValue);
                }).ToList();

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string guid = Guid.NewGuid().ToString("N");
                ReplayBuffer.WriteTextAtomic(
                    Path.Combine(MasterBuffer.DataDir, $"csv_game_{timestamp}_{guid}.json"),
                    JsonSerializer.Serialize(new MasterGameData(examples, standardizedMoves)));
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

            // 统一经 RuntimeDiagnostics 写入(每天一个文件,且与计数器日志共用一把锁,避免双开同文件冲突丢行)
            RuntimeDiagnostics.Log(msg);

            OnLog?.Invoke(msg);
        }
    }
}
