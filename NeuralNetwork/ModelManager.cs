using System;
using System.IO;
using System.Linq;

namespace ChineseChessAI.NeuralNetwork
{
    public static class ModelManager
    {
        private static readonly object _saveLock = new object();

        public static void SaveModel(CChessNet model, string filePath)
        {
            lock (_saveLock)
            {
                string? directory = Path.GetDirectoryName(filePath);
                if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                if (File.Exists(filePath))
                {
                    CreateBackup(filePath, directory ?? "");
                }

                try
                {
                    // 【关键修复】：为了防止 TorchSharp 在保存 CUDA 上的模型时崩溃，
                    // 先在 CPU 上克隆出一个副本进行保存。这解决了 Save(Tensor, BinaryWriter) 的潜在设备同步问题。
                    using (var cpuModel = new CChessNet())
                    {
                        cpuModel.load_state_dict(model.state_dict());
                        cpuModel.to(TorchSharp.DeviceType.CPU);
                        cpuModel.save(filePath);
                    }
                    Console.WriteLine($"[ModelManager] 模型参数已成功保存至: {filePath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ModelManager] 保存失败: {ex.Message}");
                    throw; // 重新抛出以供训练器捕获
                }
            }
        }

        private static void CreateBackup(string filePath, string directory)
        {
            try
            {
                // 1. 生成带时间戳的备份文件名 
                // 【核心修复】：增加 _fff 毫秒级精度，防止同一秒钟内多次触发保存导致重名
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff");
                string fileName = Path.GetFileNameWithoutExtension(filePath);
                string extension = Path.GetExtension(filePath);
                string backupPath = Path.Combine(directory, $"{fileName}_bak_{timestamp}{extension}");

                // 2. 复制当前文件到备份路径 
                // 【核心修复】：加入 overwrite: true 参数，确保即使出现极端并发也能安全覆盖而不抛出异常
                File.Copy(filePath, backupPath, overwrite: true);

                // 3. 管理备份数量：获取所有备份文件并按创建时间排序
                var backupFiles = Directory.GetFiles(directory, $"{fileName}_bak_*{extension}")
                                           .Select(f => new FileInfo(f))
                                           .OrderByDescending(f => f.CreationTime)
                                           .ToList();

                // 4. 如果超过 5 个，删除最旧的
                if (backupFiles.Count > 5)
                {
                    for (int i = 5; i < backupFiles.Count; i++)
                    {
                        backupFiles[i].Delete();
                        Console.WriteLine($"[ModelManager] 已清理旧备份: {backupFiles[i].Name}");
                    }
                }
            }
            catch (Exception ex)
            {
                // 备份失败不应中断主训练流程，仅记录日志
                Console.WriteLine($"[ModelManager] 备份失败: {ex.Message}");
            }
        }

        public static void LoadModel(CChessNet model, string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"找不到指定的模型文件: {filePath}");
            }
            model.load(filePath);
            model.eval();
            Console.WriteLine($"[ModelManager] 成功加载权重: {filePath}");
        }
    }
}