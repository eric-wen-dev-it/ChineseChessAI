namespace ChineseChessAI.Core
{
    /// <summary>
    /// 代表一个象棋动作
    /// </summary>
    public struct Move
    {
        public readonly int From; // 0-89
        public readonly int To;   // 0-89

        public Move(int from, int to)
        {
            From = from;
            To = to;
        }

        /// <summary>
        /// 将动作转换为神经网络的索引 (0-2085)
        /// cchess-zero 常用编码：FromIndex * 90 + ToIndex 后的空间压缩映射
        /// 或者直接使用 90 * 90 的简易编码（如果你内存充足且不打算完全同步原版权重）
        /// </summary>
        public int ToNetworkIndex()
        {
            // 简单实现：将 90x90 的起止点映射到索引空间
            return From * 90 + To;
        }

        /// <summary>
        /// 从网络输出的索引解析出动作
        /// </summary>
        public static Move FromNetworkIndex(int index)
        {
            return new Move(index / 90, index % 90);
        }

        /// <summary>
        /// 转换为 UCCI 协议字符串（如 "b2e2", "h0g2"）
        /// </summary>
        public override string ToString()
        {
            int fromRow = From / 9;
            int fromCol = From % 9;
            int toRow = To / 9;
            int toCol = To % 9;

            return $"{(char)('a' + fromCol)}{9 - fromRow}{(char)('a' + toCol)}{9 - toRow}";
        }
    }
}