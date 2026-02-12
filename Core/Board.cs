using System;
using System.Collections.Generic;

namespace ChineseChessAI.Core
{
    /// <summary>
    /// 代表象棋棋盘状态。
    /// 坐标系：90个位置，从左上角(0,0)到右下角(9,8)。
    /// </summary>
    public class Board
    {
        // 使用一维数组存储棋盘，方便索引
        // 0 代表空，正数代表红方，负数代表黑方
        // 1: 帅, 2: 仕, 3: 相, 4: 马, 5: 车, 6: 炮, 7: 兵
        private readonly sbyte[] _cells = new sbyte[90];

        public bool IsRedTurn { get; private set; } = true;

        public Board()
        {
            Reset();
        }

        public void Reset()
        {
            // 清空棋盘
            Array.Clear(_cells, 0, _cells.Length);

            // 定义棋子代号 (符合 cchess-zero 习惯)
            // 红方: 1帅, 2仕, 3相, 4马, 5车, 6炮, 7兵
            // 黑方: -1将, -2士, -3象, -4马, -5车, -6炮, -7卒

            // --- 摆放黑方 (第 0-4 行) ---
            _cells[0] = _cells[8] = -5; // 黑车
            _cells[1] = _cells[7] = -4; // 黑马
            _cells[2] = _cells[6] = -3; // 黑象
            _cells[3] = _cells[5] = -2; // 黑士
            _cells[4] = -1;             // 黑将

            _cells[2 * 9 + 1] = _cells[2 * 9 + 7] = -6; // 黑炮
            for (int i = 0; i < 9; i += 2)
            {
                _cells[3 * 9 + i] = -7; // 黑卒
            }

            // --- 摆放红方 (第 5-9 行) ---
            for (int i = 0; i < 9; i += 2)
            {
                _cells[6 * 9 + i] = 7;  // 红兵
            }
            _cells[7 * 9 + 1] = _cells[7 * 9 + 7] = 6;  // 红炮

            _cells[9 * 9 + 0] = _cells[9 * 9 + 8] = 5;  // 红车
            _cells[9 * 9 + 1] = _cells[9 * 9 + 7] = 4;  // 红马
            _cells[9 * 9 + 2] = _cells[9 * 9 + 6] = 3;  // 红相
            _cells[9 * 9 + 3] = _cells[9 * 9 + 5] = 2;  // 红仕
            _cells[9 * 9 + 4] = 1;                      // 红帅

            IsRedTurn = true; // 红方先行
        }

        /// <summary>
        /// 获取特定坐标的棋子
        /// </summary>
        public sbyte GetPiece(int row, int col) => _cells[row * 9 + col];

        /// <summary>
        /// 执行移动（不含合法性检查，由 MoveGenerator 处理）
        /// </summary>
        public void Push(int from, int to)
        {
            _cells[to] = _cells[from];
            _cells[from] = 0;
            IsRedTurn = !IsRedTurn;
        }

        // 获取当前棋盘的副本
        public sbyte[] GetState() => (sbyte[])_cells.Clone();
    }
}