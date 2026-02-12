using System;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.NeuralNetwork
{
    public static class ModelManager
    {
        /// <summary>
        /// 保存模型的参数 (State Dict)
        /// </summary>
        public static void SaveModel(CChessNet model, string filePath)
        {
            // 修复 CS8600: 使用 null 检查处理可能的 null 路径
            string? directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // 修复 CS1061: TorchSharp 通常使用 save 方法
            // 确保 model 继承自 torch.nn.Module
            model.save(filePath);
            Console.WriteLine($"[ModelManager] 模型参数已成功保存至: {filePath}");
        }

        /// <summary>
        /// 加载模型参数
        /// </summary>
        public static void LoadModel(CChessNet model, string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"找不到指定的模型文件: {filePath}");
            }

            // 修复 CS1061: 使用 load 加载权重
            model.load(filePath);
            model.eval(); // 加载后切换到评估模式，关闭 BatchNorm 的训练状态
            Console.WriteLine($"[ModelManager] 成功加载权重: {filePath}");
        }
    }
}