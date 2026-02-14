using System;
using System.Collections.Generic;
using System.Linq;

namespace ChineseChessAI.Core
{
    /// <summary>
    /// 存储每一步的历史状态，用于撤销和长捉/长将检测
    /// </summary>
    public record GameState(int From, int To, sbyte Captured, ulong Hash);

    public class Board
    {
        // 0 代表空，正数红方，负数黑方
        // 1:帅, 2:仕, 3:相, 4:马, 5:车, 6:炮, 7:兵
        public readonly sbyte[] _cells = new sbyte[90];
        public bool IsRedTurn { get; private set; } = true;

        // 当前局面的 Zobrist 哈希值
        public ulong CurrentHash
        {
            get; private set;
        }

        // 历史记录栈
        private readonly Stack<GameState> _history = new();

        // Zobrist 随机数表
        private static readonly ulong[,] PieceKeys = new ulong[90, 15]; // 90个位置 x 15种可能(7黑+0+7红)
        private static readonly ulong SideKey;

        static Board()
        {
            var rnd = new Random(42); // 使用固定种子确保哈希一致性
            for (int i = 0; i < 90; i++)
            {
                for (int j = 0; j < 15; j++)
                {
                    PieceKeys[i, j] = (ulong)rnd.NextInt64();
                }
            }
            SideKey = (ulong)rnd.NextInt64();
        }

        public Board()
        {
            Reset();
        }

        public void Reset()
        {
            Array.Clear(_cells, 0, _cells.Length);

            // --- 摆放黑方 (第 0-3 行) ---
            _cells[0] = _cells[8] = -5; // 黑车
            _cells[1] = _cells[7] = -4; // 黑马
            _cells[2] = _cells[6] = -3; // 黑象
            _cells[3] = _cells[5] = -2; // 黑士
            _cells[4] = -1;             // 黑将
            _cells[19] = _cells[25] = -6; // 黑炮
            for (int i = 0; i < 9; i += 2)
                _cells[27 + i] = -7; // 黑卒

            // --- 摆放红方 (第 6-9 行) ---
            for (int i = 0; i < 9; i += 2)
                _cells[6 * 9 + i] = 7;  // 红兵
            _cells[7 * 9 + 1] = _cells[7 * 9 + 7] = 6;            // 红炮
            _cells[9 * 9 + 0] = _cells[9 * 9 + 8] = 5; // 红车
            _cells[9 * 9 + 1] = _cells[9 * 9 + 7] = 4; // 红马
            _cells[9 * 9 + 2] = _cells[9 * 9 + 6] = 3; // 红相
            _cells[9 * 9 + 3] = _cells[9 * 9 + 5] = 2; // 红仕
            _cells[9 * 9 + 4] = 1;                     // 红帅

            IsRedTurn = true;
            _history.Clear();
            CalculateFullHash();
        }

        public sbyte GetPiece(int row, int col) => _cells[row * 9 + col];
        public sbyte GetPiece(int index) => _cells[index];

        /// <summary>
        /// 标准走棋：更新棋盘、切换回合、记录历史并更新哈希
        /// </summary>
        public void Push(int from, int to)
        {
            sbyte piece = _cells[from];
            sbyte captured = _cells[to];

            // 1. 记录当前状态
            _history.Push(new GameState(from, to, captured, CurrentHash));

            // 2. 更新哈希（增量更新：异或掉旧位置，异或上新位置）
            TogglePieceHash(from, piece);      // 移除起点棋子
            if (captured != 0)
                TogglePieceHash(to, captured); // 移除被吃掉的棋子
            TogglePieceHash(to, piece);        // 在终点放入棋子
            CurrentHash ^= SideKey;            // 切换回合哈希

            // 3. 物理移动
            _cells[to] = piece;
            _cells[from] = 0;
            IsRedTurn = !IsRedTurn;
        }

        /// <summary>
        /// 撤销上一步走法
        /// </summary>
        public void Pop()
        {
            if (_history.Count == 0)
                return;

            var last = _history.Pop();

            // 物理恢复
            _cells[last.From] = _cells[last.To];
            _cells[last.To] = last.Captured;
            IsRedTurn = !IsRedTurn;

            // 恢复哈希
            CurrentHash = last.Hash;
        }

        /// <summary>
        /// 检测当前局面在历史中出现的次数
        /// </summary>
        public int GetRepetitionCount()
        {
            int count = 1;
            foreach (var state in _history)
            {
                if (state.Hash == CurrentHash)
                    count++;
            }
            return count;
        }

        /// <summary>
        /// 获取完整的历史记录（用于 MoveGenerator 分析长捉属性）
        /// </summary>
        public IEnumerable<GameState> GetHistory() => _history;

        // --- 合法性检查专用辅助方法（不改变哈希和历史） ---

        public sbyte PerformMoveInternal(int from, int to)
        {
            sbyte captured = _cells[to];
            _cells[to] = _cells[from];
            _cells[from] = 0;
            return captured;
        }

        public void UndoMoveInternal(int from, int to, sbyte captured)
        {
            _cells[from] = _cells[to];
            _cells[to] = captured;
        }

        // --- 内部哈希计算逻辑 ---

        private void TogglePieceHash(int pos, sbyte piece)
        {
            // piece + 7 将 [-7, 7] 映射到 [0, 14]
            CurrentHash ^= PieceKeys[pos, piece + 7];
        }

        private void CalculateFullHash()
        {
            CurrentHash = 0;
            for (int i = 0; i < 90; i++)
            {
                if (_cells[i] != 0)
                    TogglePieceHash(i, _cells[i]);
            }
            if (IsRedTurn)
                CurrentHash ^= SideKey;
        }

        // --- 状态快照 ---

        public sbyte[] GetState() => (sbyte[])_cells.Clone();

        public void LoadState(sbyte[] state, bool isRedTurn)
        {
            Array.Copy(state, _cells, 90);
            this.IsRedTurn = isRedTurn;
            CalculateFullHash();
            _history.Clear();
        }



        // 修改原有的 GetMoveHistoryString 方法以显示新格式
        public string GetMoveHistoryString()
        {
            // 需要通过模拟走法来正确获取每一步执行前的棋子位置
            var tempBoard = new Board();
            var history = _history.Reverse().ToList();
            var result = new List<string>();

            foreach (var state in history)
            {
                result.Add(tempBoard.GetChineseMoveName(state.From, state.To));
                tempBoard.Push(state.From, state.To);
            }

            return string.Join(" ", result);
        }

        // 新增：带步数编号的易读格式（用于 UI 显示）
        public string GetReadableMoveHistory()
        {
            var moves = _history.Select(s => new Move(s.From, s.To)).Reverse().ToList();
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < moves.Count; i++)
            {
                if (i % 2 == 0)
                    sb.Append($"{i / 2 + 1}. ");
                sb.Append($"{moves[i]} ");
            }
            return sb.ToString().Trim();
        }

        // Core/Board.cs

        /// <summary>
        /// 获取单步走法的中文标准表示（如：兵1进1）
        /// </summary>
        public string GetChineseMoveName(int from, int to)
        {
            sbyte piece = _cells[from];
            if (piece == 0)
                return "";

            bool isRed = piece > 0;
            int fromR = from / 9;
            int fromC = from % 9;
            int toR = to / 9;
            int toC = to % 9;

            // 1. 获取棋子名称
            string name = GetPieceName(piece);

            // 2. 计算原始纵线 (1-9)
            // 红方：右(9)->左(0) 映射为 1-9
            // 黑方：左(0)->右(9) 映射为 1-9
            int fromCol = isRed ? (9 - fromC) : (fromC + 1);
            int toCol = isRed ? (9 - toC) : (toC + 1);

            // 3. 确定动作
            string action = "";
            if (toR == fromR)
                action = "平";
            else if (isRed)
                action = (toR < fromR) ? "进" : "退";
            else
                action = (toR > fromR) ? "进" : "退";

            // 4. 计算目标值 (纵线号或距离)
            int targetValue;
            int type = Math.Abs(piece);

            // 马(4)、象(3)、士(2) 永远走斜线，最后一位是目标纵线
            if (type == 2 || type == 3 || type == 4)
            {
                targetValue = toCol;
            }
            else // 车、炮、兵、将
            {
                if (action == "平")
                    targetValue = toCol;
                else
                    targetValue = Math.Abs(toR - fromR); // 进退则记录步数
            }

            return $"{name}{fromCol}{action}{targetValue}";
        }

        private string GetPieceName(sbyte p)
        {
            string[] namesRed = { "", "帅", "仕", "相", "马", "车", "炮", "兵" };
            string[] namesBlack = { "", "将", "士", "象", "马", "车", "炮", "卒" };
            return p > 0 ? namesRed[p] : namesBlack[-p];
        }

       
    }
}