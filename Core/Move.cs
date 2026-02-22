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
        /// 将动作转换为神经网络的索引 (0-8099)
        /// 采用 90 * 90 映射。注意：训练时黑方视角必须先进行 180 度翻转再计算索引。
        /// </summary>
        public int ToNetworkIndex()
        {
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
        /// 转换为标准的 UCCI 协议字符串（如 "a9a8", "b2e2"）
        /// 修正：确保无论红黑方，a-i 始终对应从左往右，9-0 始终对应从上往下。
        /// </summary>
        public override string ToString()
        {
            // 棋盘布局（红方视角）：
            // 行：0(最顶端/黑方底线) - 9(最底端/红方底线)
            // 列：0(最左侧/a路) - 8(最右侧/i路)

            int fromRow = From / 9;
            int fromCol = From % 9;
            int toRow = To / 9;
            int toCol = To % 9;

            // 检查非法动作（原地不动）
            if (From == To)
            {
                return "null";
            }

            // UCCI 坐标：
            // 列：'a' + colIndex
            // 行：9 - rowIndex (WPF 坐标 0 行对应 UCCI 的 9 行)
            char fC = (char)('a' + fromCol);
            int fR = 9 - fromRow;
            char tC = (char)('a' + toCol);
            int tR = 9 - toRow;

            return $"{fC}{fR}{tC}{tR}";
        }
    }
}