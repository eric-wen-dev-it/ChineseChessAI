namespace ChineseChessAI.Play
{
    public sealed class PlayStrengthSettings
    {
        public int DefaultSimulations { get; set; } = 2400;
        public int BatchSize { get; set; } = 64;
        public double CPuct { get; set; } = 1.6;
        public bool AddRootNoise { get; set; } = false;

        public static PlayStrengthSettings Sanitize(PlayStrengthSettings? settings)
        {
            settings ??= new PlayStrengthSettings();

            if (settings.DefaultSimulations <= 0)
                settings.DefaultSimulations = 2400;
            if (settings.BatchSize <= 0)
                settings.BatchSize = 64;
            if (settings.CPuct <= 0)
                settings.CPuct = 1.6;

            return settings;
        }
    }
}
