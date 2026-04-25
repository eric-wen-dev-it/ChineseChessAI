namespace ChineseChessAI.Core
{
    public record struct ActionProb(int Index, float Prob);

    public record TrainingExample(float[] State, ActionProb[] SparsePolicy, float Value);

    public record MasterGameData
    {
        public List<TrainingExample> Examples { get; init; } = new List<TrainingExample>();
        public List<string> MoveHistoryUcci { get; init; } = new List<string>();
        public DateTimeOffset? StartedAt
        {
            get; init;
        }
        public DateTimeOffset? EndedAt
        {
            get; init;
        }
        public TimeSpan? Elapsed
        {
            get; init;
        }
        public string? Result
        {
            get; init;
        }
        public string? EndReason
        {
            get; init;
        }
        public int? MoveCount
        {
            get; init;
        }
        public int? GameId
        {
            get; init;
        }

        public MasterGameData()
        {
        }

        public MasterGameData(List<TrainingExample> examples, List<string> moveHistoryUcci)
        {
            Examples = examples;
            MoveHistoryUcci = moveHistoryUcci;
        }
    }
}
