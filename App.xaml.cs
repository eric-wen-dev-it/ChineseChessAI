using System;
using System.Windows;

namespace ChineseChessAI
{
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            // 在 CUDA 上下文初始化之前设置，解决多次迭代后显存碎片化导致的 OOM 问题
            Environment.SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
            base.OnStartup(e);
        }
    }
}
