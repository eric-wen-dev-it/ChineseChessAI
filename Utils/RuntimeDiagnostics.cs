using System.IO;
using System.Text;

namespace ChineseChessAI.Utils
{
    internal static class RuntimeDiagnostics
    {
        private static readonly object LogLock = new object();
        private static readonly string RuntimeLogPath = Path.Combine(
            AppDomain.CurrentDomain.BaseDirectory,
            "data",
            "runtime.log");

        public static void Log(string message)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(RuntimeLogPath)!);
                string line = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} {message}{Environment.NewLine}";
                lock (LogLock)
                {
                    File.AppendAllText(RuntimeLogPath, line, Encoding.UTF8);
                }
            }
            catch
            {
            }
        }

        internal sealed class RollingCounter
        {
            private readonly string _name;
            private readonly int _emitEvery;
            private long _count;
            private long _sum;
            private long _max;

            public RollingCounter(string name, int emitEvery)
            {
                _name = name;
                _emitEvery = emitEvery;
            }

            public void AddSample(long value)
            {
                long count = Interlocked.Increment(ref _count);
                Interlocked.Add(ref _sum, value);

                while (true)
                {
                    long currentMax = Volatile.Read(ref _max);
                    if (value <= currentMax)
                    {
                        break;
                    }

                    if (Interlocked.CompareExchange(ref _max, value, currentMax) == currentMax)
                    {
                        break;
                    }
                }

                if (count % _emitEvery == 0)
                {
                    long sum = Volatile.Read(ref _sum);
                    long max = Volatile.Read(ref _max);
                    double avg = count > 0 ? (double)sum / count : 0.0;
                    Log($"[{_name}] samples={count} avg={avg:F2} max={max}");
                }
            }
        }
    }
}
