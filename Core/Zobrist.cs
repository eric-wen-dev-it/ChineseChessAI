using System;

namespace ChineseChessAI.Core
{
    public static class Zobrist
    {
        // 为 90 个位置的每种棋子（14种：红黑各7种）生成随机数
        private static readonly ulong[,] PieceKeys = new ulong[90, 15];
        private static readonly ulong SideKey;

        static Zobrist()
        {
            var rnd = new Random(42);
            for (int i = 0; i < 90; i++)
                for (int j = 0; j < 15; j++)
                    PieceKeys[i, j] = (ulong)rnd.NextInt64();
            SideKey = (ulong)rnd.NextInt64();
        }

        public static ulong Calculate(sbyte[] cells, bool isRedTurn)
        {
            ulong h = 0;
            for (int i = 0; i < 90; i++)
            {
                if (cells[i] != 0)
                {
                    // 将 sbyte (-7 to 7) 映射到 0-14
                    int pieceIdx = cells[i] + 7;
                    h ^= PieceKeys[i, pieceIdx];
                }
            }
            if (isRedTurn)
                h ^= SideKey;
            return h;
        }
    }
}