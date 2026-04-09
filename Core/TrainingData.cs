using System.Collections.Generic;

namespace ChineseChessAI.Core
{
    public record struct ActionProb(int Index, float Prob);

    public record TrainingExample(float[] State, ActionProb[] SparsePolicy, float Value);

    public record MasterGameData(List<TrainingExample> Examples, List<string> MoveHistoryUcci);
}
