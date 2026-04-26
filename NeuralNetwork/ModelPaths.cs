using System.IO;

namespace ChineseChessAI.NeuralNetwork
{
    public static class ModelPaths
    {
        public const string BestModelsDirectoryName = "best";
        public const string BestModelFileName = "best_model.pt";

        public static string GetModelsDirectory(string root)
        {
            return Path.Combine(root, "data", "models");
        }

        public static string GetBestModelsDirectory(string root)
        {
            return Path.Combine(GetModelsDirectory(root), BestModelsDirectoryName);
        }

        public static string GetBestModelPath(string root)
        {
            return Path.Combine(GetBestModelsDirectory(root), BestModelFileName);
        }

        public static void EnsureBestModelsDirectory(string root)
        {
            Directory.CreateDirectory(GetBestModelsDirectory(root));
        }
    }
}
