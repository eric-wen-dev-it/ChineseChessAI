using System;
using System.Collections.Generic;
using System.Linq;

namespace ChineseChessAI.Core
{
    /// <summary>
    /// 存储每一步的历史状态，用于撤销和长捉/长将检测
    /// </summary>
    public record GameState(int From, int To, sbyte Captured, ulong Hash, Move? LastMoveBefore);

    public class Board
    {
        // 0 代表空，正数红方，负数黑方
        // 1:帅, 2:仕, 3:相, 4:马, 5:车, 6:炮, 7:兵
        public readonly sbyte[] _cells = new sbyte[90];
        public bool IsRedTurn { get; private set; } = true;

        // 存储最后一步棋的起始和结束位置，供 UI 绘制红/绿方框
        public Move? LastMove
        {
            get; private set;
        }

        // 当前局面的 Zobrist 哈希值
        public ulong CurrentHash
        {
            get; private set;
        }

        // 历史记录栈
        private readonly Stack<GameState> _history = new();

        // Zobrist 随机数表
        private static readonly ulong[,] PieceKeys = new ulong[90, 15];
        private static readonly ulong SideKey;

        static Board()
        {
            var rnd = new Random(42);
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
            LastMove = null;
            CalculateFullHash();
        }

        public sbyte GetPiece(int row, int col) => _cells[row * 9 + col];
        public sbyte GetPiece(int index) => _cells[index];

        public bool WillCauseThreefoldRepetition(int from, int to)
        {
            sbyte piece = _cells[from];
            sbyte captured = _cells[to];
            ulong nextHash = CurrentHash;

            nextHash ^= PieceKeys[from, piece + 7];
            if (captured != 0)
                nextHash ^= PieceKeys[to, captured + 7];
            nextHash ^= PieceKeys[to, piece + 7];
            nextHash ^= SideKey;

            int count = _history.Count(s => s.Hash == nextHash);
            return count >= 2;
        }

        public void Push(int from, int to)
        {
            sbyte piece = _cells[from];
            sbyte captured = _cells[to];

            _history.Push(new GameState(from, to, captured, CurrentHash, LastMove));
            LastMove = new Move(from, to);

            TogglePieceHash(from, piece);
            if (captured != 0)
                TogglePieceHash(to, captured);
            TogglePieceHash(to, piece);
            CurrentHash ^= SideKey;

            _cells[to] = piece;
            _cells[from] = 0;
            IsRedTurn = !IsRedTurn;
        }

        public void Pop()
        {
            if (_history.Count == 0)
                return;
            var last = _history.Pop();
            _cells[last.From] = _cells[last.To];
            _cells[last.To] = last.Captured;
            IsRedTurn = !IsRedTurn;
            CurrentHash = last.Hash;
            LastMove = last.LastMoveBefore;
        }

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

        public IEnumerable<GameState> GetHistory() => _history;

        private void TogglePieceHash(int pos, sbyte piece)
        {
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
            if (!IsRedTurn)
                CurrentHash ^= SideKey;
        }

        // --- 核心修复：走法文本转换 ---

        public string GetChineseMoveName(int from, int to)
        {
            if (from == to)
                return "原地"; // 异常防御

            sbyte piece = _cells[from];
            if (piece == 0)
                return "";

            bool isRed = piece > 0;
            int fromR = from / 9, fromC = from % 9;
            int toR = to / 9, toC = to % 9;

            string name = GetPieceName(piece);

            // 【标准规则】路数计算
            // 红方（底线在下）：从右往左数 1-9 (i列是1，a列是9) -> 公式: 9 - colIndex
            // 黑方（底线在上）：从左往右数 1-9 (a列是1，i列是9) -> 公式: colIndex + 1
            int fromCol = isRed ? (9 - fromC) : (fromC + 1);
            int toCol = isRed ? (9 - toC) : (toC + 1);

            // 动作判定
            string action;
            if (toR == fromR)
                action = "平";
            else if (isRed)
                action = (toR < fromR) ? "进" : "退"; // 红方向上为进
            else
                action = (toR > fromR) ? "进" : "退";             // 黑方向下为进

            // 目标数值计算
            int targetValue;
            int type = Math.Abs(piece);

            // 士、相、马：斜走棋子，终点永远记录目标路数
            if (type == 2 || type == 3 || type == 4)
            {
                targetValue = toCol;
            }
            else
            {
                // 车、炮、兵、帅：平移动作记路数，进退动作记步数
                targetValue = (action == "平") ? toCol : Math.Abs(toR - fromR);
            }

            return $"{name}{fromCol}{action}{targetValue}";
        }

        public static string GetPieceName(sbyte p)
        {
            string[] namesRed = { "", "帅", "仕", "相", "傌", "俥", "炮", "兵" };
            string[] namesBlack = { "", "将", "士", "象", "馬", "車", "砲", "卒" };
            return p > 0 ? namesRed[p] : namesBlack[-p];
        }

        public string GetMoveHistoryString()
        {
            var tempBoard = new Board();
            var historyList = _history.Reverse().ToList();
            var result = new List<string>();
            foreach (var state in historyList)
            {
                // 必须在执行 Push 之前获取名称，因为名称依赖起始位置的棋子类型
                result.Add(tempBoard.GetChineseMoveName(state.From, state.To));
                tempBoard.Push(state.From, state.To);
            }
            return string.Join(" ", result);
        }

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

        public sbyte[] GetState() => (sbyte[])_cells.Clone();

        public void LoadState(sbyte[] state, bool isRedTurn)
        {
            Array.Copy(state, _cells, 90);
            this.IsRedTurn = isRedTurn;
            CalculateFullHash();
            _history.Clear();
            LastMove = null;
        }

        public void RecordHistory(GameState state)
        {
            _history.Push(state);
            LastMove = new Move(state.From, state.To);
        }
    }
}