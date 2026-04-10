using ChineseChessAI.Training;
using System;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

namespace ChineseChessAI
{
    public partial class App : Application
    {
        private static readonly object CrashLogLock = new object();
        private static readonly string CrashLogPath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "crash.log");

        private bool _suppressErrorDialogs;

        protected override async void OnStartup(StartupEventArgs e)
        {
            // Set before CUDA context initialization to reduce allocator fragmentation.
            Environment.SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");

            Directory.CreateDirectory(Path.GetDirectoryName(CrashLogPath)!);
            DispatcherUnhandledException += OnDispatcherUnhandledException;
            AppDomain.CurrentDomain.UnhandledException += OnAppDomainUnhandledException;
            TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;

            base.OnStartup(e);

            if (TryParseHeadlessLeagueOptions(e.Args, out var options))
            {
                _suppressErrorDialogs = true;
                ShutdownMode = ShutdownMode.OnExplicitShutdown;
                await RunHeadlessLeagueAsync(options);
                return;
            }

            ShutdownMode = ShutdownMode.OnMainWindowClose;
            MainWindow = new MainWindow();
            MainWindow.Show();
        }

        private async Task RunHeadlessLeagueAsync(HeadlessLeagueOptions options)
        {
            var orchestrator = new TrainingOrchestrator();
            var trainingStopped = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);

            orchestrator.OnTrainingStopped += () => trainingStopped.TrySetResult(true);
            orchestrator.OnError += err => AppendCrashLog("HeadlessLeagueError", new Exception(err));

            try
            {
                await orchestrator.StartLeagueTrainingAsync(
                    populationSize: options.PopulationSize,
                    maxMoves: options.MaxMoves,
                    exploreMoves: options.ExploreMoves,
                    materialBias: options.MaterialBias,
                    populationRefreshInterval: options.RefreshInterval,
                    maxPopulationRefreshCycles: options.RefreshCycles);

                await trainingStopped.Task;
                Shutdown(0);
            }
            catch (Exception ex)
            {
                AppendCrashLog("HeadlessLeagueFatal", ex);
                Shutdown(1);
            }
        }

        private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
        {
            AppendCrashLog("DispatcherUnhandledException", e.Exception);
            if (!_suppressErrorDialogs)
            {
                MessageBox.Show(
                    $"绋嬪簭鎹曡幏鍒版湭澶勭悊寮傚父锛岃鎯呭凡鍐欏叆锛歕n{CrashLogPath}",
                    "杩愯寮傚父",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error);
            }

            e.Handled = true;
        }

        private void OnAppDomainUnhandledException(object? sender, UnhandledExceptionEventArgs e)
        {
            if (e.ExceptionObject is Exception ex)
            {
                AppendCrashLog("AppDomainUnhandledException", ex);
            }
            else
            {
                AppendCrashLog("AppDomainUnhandledException", new Exception(e.ExceptionObject?.ToString() ?? "Unknown fatal error."));
            }
        }

        private void OnUnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
        {
            AppendCrashLog("UnobservedTaskException", e.Exception);
            e.SetObserved();
        }

        private static bool TryParseHeadlessLeagueOptions(string[] args, out HeadlessLeagueOptions options)
        {
            options = default;

            if (!args.Contains("--headless-league", StringComparer.OrdinalIgnoreCase))
            {
                return false;
            }

            int populationSize = 50;
            int refreshInterval = 12;
            int refreshCycles = 3;
            int maxMoves = 150;
            int exploreMoves = 40;
            float materialBias = 0.1f;

            foreach (string arg in args)
            {
                if (TryGetIntArg(arg, "--population=", out int parsedPopulation))
                {
                    populationSize = parsedPopulation;
                }
                else if (TryGetIntArg(arg, "--refresh-interval=", out int parsedInterval))
                {
                    refreshInterval = parsedInterval;
                }
                else if (TryGetIntArg(arg, "--refresh-cycles=", out int parsedCycles))
                {
                    refreshCycles = parsedCycles;
                }
                else if (TryGetIntArg(arg, "--max-moves=", out int parsedMaxMoves))
                {
                    maxMoves = parsedMaxMoves;
                }
                else if (TryGetIntArg(arg, "--explore-moves=", out int parsedExploreMoves))
                {
                    exploreMoves = parsedExploreMoves;
                }
                else if (TryGetFloatArg(arg, "--material-bias=", out float parsedMaterialBias))
                {
                    materialBias = parsedMaterialBias;
                }
            }

            options = new HeadlessLeagueOptions(
                PopulationSize: populationSize,
                RefreshInterval: refreshInterval,
                RefreshCycles: refreshCycles,
                MaxMoves: maxMoves,
                ExploreMoves: exploreMoves,
                MaterialBias: materialBias);
            return true;
        }

        private static bool TryGetIntArg(string arg, string prefix, out int value)
        {
            value = 0;
            if (!arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            return int.TryParse(arg[prefix.Length..], NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryGetFloatArg(string arg, string prefix, out float value)
        {
            value = 0;
            if (!arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            return float.TryParse(arg[prefix.Length..], NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        private static void AppendCrashLog(string source, Exception ex)
        {
            string content =
                $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] {source}{Environment.NewLine}" +
                $"{ex}{Environment.NewLine}" +
                new string('-', 80) +
                Environment.NewLine;

            lock (CrashLogLock)
            {
                File.AppendAllText(CrashLogPath, content);
            }
        }

        private readonly record struct HeadlessLeagueOptions(
            int PopulationSize,
            int RefreshInterval,
            int RefreshCycles,
            int MaxMoves,
            int ExploreMoves,
            float MaterialBias);
    }
}
