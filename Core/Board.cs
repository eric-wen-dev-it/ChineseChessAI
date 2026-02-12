using System;

namespace ChineseChessAI.Core
{
    public class Board
    {
        // 0 代表空，正数红方，负数黑方
        private readonly sbyte[] _cells = new sbyte[90];
        public bool IsRedTurn { get; private set; } = true;

        public Board()
        {
            Reset();
        }

        /// <summary>
        /// 修复后的棋盘初始化
        /// </summary>
        public void Reset()
        {
            Array.Clear(_cells, 0, _cells.Length);

            // --- 摆放黑方 (第 0-3 行) ---
            _cells[0] = _cells[8] = -5; // 黑车
            _cells[1] = _cells[7] = -4; // 黑马
            _cells[2] = _cells[6] = -3; // 黑象
            _cells[3] = _cells[5] = -2; // 黑士
            _cells[4] = -1;             // 黑将

            _cells[2 * 9 + 1] = _cells[2 * 9 + 7] = -6; // 黑炮
            for (int i = 0; i < 9; i += 2)
                _cells[3 * 9 + i] = -7; // 黑卒

            // --- 摆放红方 (第 6-9 行) ---
            for (int i = 0; i < 9; i += 2)
                _cells[6 * 9 + i] = 7;  // 红兵
            _cells[7 * 9 + 1] = _cells[7 * 9 + 7] = 6;            // 红炮

            // 修复点：确保索引严格计算，不要使用容易混淆的加法
            _cells[9 * 9 + 0] = _cells[9 * 9 + 8] = 5; // 红车 (索引 81, 89)
            _cells[9 * 9 + 1] = _cells[9 * 9 + 7] = 4; // 红马 (索引 82, 88)
            _cells[9 * 9 + 2] = _cells[9 * 9 + 6] = 3; // 红相 (索引 83, 87)
            _cells[9 * 9 + 3] = _cells[9 * 9 + 5] = 2; // 红仕 (索引 84, 86)
            _cells[9 * 9 + 4] = 1;                     // 红帅 (索引 85)

            IsRedTurn = true;
        }

        public sbyte GetPiece(int row, int col) => _cells[row * 9 + col];

        public void Push(int from, int to)
        {
            if (from < 0 || from >= 90 || to < 0 || to >= 90)
                return; // 安全检查
            _cells[to] = _cells[from];
            _cells[from] = 0;
            IsRedTurn = !IsRedTurn;
        }

        // 必须提供状态快照以便 MCTS 克隆
        public sbyte[] GetState() => (sbyte[])_cells.Clone();

        /// <summary>
        /// 供 MCTSEngine 使用的同步方法
        /// </summary>
        public void LoadState(sbyte[] state, bool isRedTurn)
        {
            Array.Copy(state, _cells, 90);
            this.IsRedTurn = isRedTurn;
        }
    }
}