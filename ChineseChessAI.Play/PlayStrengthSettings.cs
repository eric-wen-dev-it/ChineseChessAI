namespace ChineseChessAI.Play
{
    public sealed class PlayStrengthSettings
    {
        public int DefaultSimulations { get; set; } = 2400;
        public int TraditionalDepth { get; set; } = 5;
        public int TraditionalMoveTimeMs { get; set; } = 5000;
        public int TraditionalRootParallelism { get; set; } = 0;
        public string PikafishPath { get; set; } = string.Empty;
        public int PikafishMoveTimeMs { get; set; } = 3000;
        public int BatchSize { get; set; } = 64;
        public double CPuct { get; set; } = 1.6;
        public bool AddRootNoise { get; set; } = false;

        public static PlayStrengthSettings Sanitize(PlayStrengthSettings? settings)
        {
            settings ??= new PlayStrengthSettings();

            if (settings.DefaultSimulations <= 0)
                settings.DefaultSimulations = 2400;
            if (settings.TraditionalDepth <= 0)
                settings.TraditionalDepth = 5;
            settings.TraditionalDepth = Math.Clamp(settings.TraditionalDepth, 1, 12);
            if (settings.TraditionalMoveTimeMs <= 0)
                settings.TraditionalMoveTimeMs = 5000;
            settings.TraditionalMoveTimeMs = Math.Clamp(settings.TraditionalMoveTimeMs, 500, 60000);
            settings.TraditionalRootParallelism = settings.TraditionalRootParallelism <= 0
                ? 0
                : Math.Clamp(settings.TraditionalRootParallelism, 1, 32);
            if (settings.PikafishMoveTimeMs <= 0)
                settings.PikafishMoveTimeMs = 3000;
            settings.PikafishMoveTimeMs = Math.Clamp(settings.PikafishMoveTimeMs, 200, 60000);
            if (settings.BatchSize <= 0)
                settings.BatchSize = 64;
            if (settings.CPuct <= 0)
                settings.CPuct = 1.6;

            return settings;
        }
    }
}
