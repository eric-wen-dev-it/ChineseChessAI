using TorchSharp;

namespace ChineseChessAI.Utils
{
    internal static class GpuExecutionGate
    {
        private static readonly SemaphoreSlim Gate = new SemaphoreSlim(1, 1);

        public static void Run(Action action)
        {
            if (!torch.cuda.is_available())
            {
                action();
                return;
            }

            Gate.Wait();
            try
            {
                action();
            }
            finally
            {
                Gate.Release();
            }
        }

        public static T Run<T>(Func<T> action)
        {
            if (!torch.cuda.is_available())
            {
                return action();
            }

            Gate.Wait();
            try
            {
                return action();
            }
            finally
            {
                Gate.Release();
            }
        }
    }
}
