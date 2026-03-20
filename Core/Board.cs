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

        public Move? LastMove
        {
            get; private set;
        }
        public ulong CurrentHash
        {
            get; private set;
        }

        private readonly Stack<GameState> _history = new();

        // 【核心提速修复】：增加 O(1) 的哈希计数表，彻底消灭历史遍历的性能黑洞
        private readonly Dictionary<ulong, int> _hashCounts = new();

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
            _cells[7 * 9 + 1] = _cells[7 * 9 + 7] = 6; // 红炮
            _cells[9 * 9 + 0] = _cells[9 * 9 + 8] = 5; // 红车
            _cells[9 * 9 + 1] = _cells[9 * 9 + 7] = 4; // 红马
            _cells[9 * 9 + 2] = _cells[9 * 9 + 6] = 3; // 红相
            _cells[9 * 9 + 3] = _cells[9 * 9 + 5] = 2; // 红仕
            _cells[9 * 9 + 4] = 1;                     // 红帅

            IsRedTurn = true;
            _history.Clear();
            _hashCounts.Clear(); // 清空计数表
            LastMove = null;

            CalculateFullHash();
            _hashCounts[CurrentHash] = 1; // 初始局面计数为 1
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

            // 【核心提速修复】：O(1) 字典瞬间查询，替换原有的 O(n) LINQ 遍历
            _hashCounts.TryGetValue(nextHash, out int count);
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

            // 【核心提速修复】：走子后，将新局面的哈希计数 O(1) 增加
            if (!_hashCounts.TryGetValue(CurrentHash, out int c))
                c = 0;
            _hashCounts[CurrentHash] = c + 1;
        }

        public void Pop()
        {
            if (_history.Count == 0)
                return;

            // 【核心提速修复】：撤销前，将当前局面的哈希计数 O(1) 减少
            if (_hashCounts.TryGetValue(CurrentHash, out int c))
            {
                if (c <= 1)
                    _hashCounts.Remove(CurrentHash);
                else
                    _hashCounts[CurrentHash] = c - 1;
            }

            var last = _history.Pop();
            _cells[last.From] = _cells[last.To];
            _cells[last.To] = last.Captured;
            IsRedTurn = !IsRedTurn;
            CurrentHash = last.Hash; // 恢复旧局面的 Hash
            LastMove = last.LastMoveBefore;
        }

        public int GetRepetitionCount()
        {
            // 【核心提速修复】：O(1) 字典查询
            _hashCounts.TryGetValue(CurrentHash, out int count);
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

        public string GetChineseMoveName(int from, int to)
        {
            if (from == to)
                return "原地";

            sbyte piece = _cells[from];
            if (piece == 0)
                return "";

            bool isRed = piece > 0;
            int fromR = from / 9, fromC = from % 9;
            int toR = to / 9, toC = to % 9;

            string name = GetPieceName(piece);

            int fromCol = isRed ? (9 - fromC) : (fromC + 1);
            int toCol = isRed ? (9 - toC) : (toC + 1);

            string action;
            if (toR == fromR)
                action = "平";
            else if (isRed)
                action = (toR < fromR) ? "进" : "退";
            else
                action = (toR > fromR) ? "进" : "退";

            int targetValue;
            int type = Math.Abs(piece);

            if (type == 2 || type == 3 || type == 4)
            {
                targetValue = toCol;
            }
            else
            {
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
            _hashCounts.Clear();
            _hashCounts[CurrentHash] = 1; // 载入新局面时初始化计数
            LastMove = null;
        }

        public void RecordHistory(GameState state)
        {
            _history.Push(state);
            // 手动推入历史时，也要同步维护哈希表
            if (!_hashCounts.TryGetValue(state.Hash, out int c))
                c = 0;
            _hashCounts[state.Hash] = c + 1;

            LastMove = new Move(state.From, state.To);
        }
    }
}