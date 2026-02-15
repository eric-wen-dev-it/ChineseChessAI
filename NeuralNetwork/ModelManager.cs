using System;
using System.IO;
using System.Linq; // 新增：用于处理列表排序
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.NeuralNetwork
{
    public static class ModelManager
    {
        public static void SaveModel(CChessNet model, string filePath)
        {
            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // --- 新增：备份逻辑开始 ---
            if (File.Exists(filePath))
            {
                CreateBackup(filePath, directory ?? "");
            }
            // --- 新增：备份逻辑结束 ---

            model.save(filePath);
            Console.WriteLine($"[ModelManager] 模型参数已成功保存至: {filePath}");
        }

        private static void CreateBackup(string filePath, string directory)
        {
            try
            {
                // 1. 生成带时间戳的备份文件名
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string fileName = Path.GetFileNameWithoutExtension(filePath);
                string extension = Path.GetExtension(filePath);
                string backupPath = Path.Combine(directory, $"{fileName}_bak_{timestamp}{extension}");

                // 2. 复制当前文件到备份路径
                File.Copy(filePath, backupPath);

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