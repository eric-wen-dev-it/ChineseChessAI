namespace ChineseChessAI.Core
{
    public static class BoardEvaluation
    {
        // 训练目标与 GUI 评估器对齐：三循/百步无进展/步数限制时，
        // 子力差超过此阈值则按材料裁判胜负，否则视为真正平局。
        public const float MaterialAdjudicationThreshold = 1.5f;

        public static float CalculateMaterialScore(Board board, bool isRed) => isRed ? board.RedMaterial : board.BlackMaterial;

        public static float GetBoardAdvantage(Board board)
        {
            float diff = board.RedMaterial - board.BlackMaterial;
            return diff > 0.5f ? 1.0f : (diff < -0.5f ? -1.0f : 0.0f);
        }

        public static float AdjudicateDrawByMaterial(Board board)
        {
            float diff = board.RedMaterial - board.BlackMaterial;
            if (diff >= MaterialAdjudicationThreshold)
                return 1.0f;
            if (diff <= -MaterialAdjudicationThreshold)
                return -1.0f;
            return 0.0f;
        }
    }
}
