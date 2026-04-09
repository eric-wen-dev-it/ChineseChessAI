namespace ChineseChessAI.Core
{
    public static class BoardEvaluation
    {
        public static float CalculateMaterialScore(Board board, bool isRed) => isRed ? board.RedMaterial : board.BlackMaterial;

        public static float GetBoardAdvantage(Board board)
        {
            float diff = board.RedMaterial - board.BlackMaterial;
            return diff > 0.5f ? 1.0f : (diff < -0.5f ? -1.0f : 0.0f);
        }
    }
}
