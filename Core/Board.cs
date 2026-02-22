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

        // 【新增】存储最后一步棋的起始和结束位置
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
            LastMove = null; // 【重置】最后一步为空
            CalculateFullHash();
        }

        public sbyte GetPiece(int row, int col) => _cells[row * 9 + col];
        public sbyte GetPiece(int index) => _cells[index];

        /// <summary>
        /// 核心：检测如果执行某一步，是否会导致三复局面
        /// </summary>
        public bool WillCauseThreefoldRepetition(int from, int to)
        {
            // 预测哈希变化
            sbyte piece = _cells[from];
            sbyte captured = _cells[to];
            ulong nextHash = CurrentHash;

            // 增量模拟哈希
            nextHash ^= PieceKeys[from, piece + 7];      // 移除起点
            if (captured != 0)
                nextHash ^= PieceKeys[to, captured + 7]; // 移除被吃子
            nextHash ^= PieceKeys[to, piece + 7];        // 放入终点
            nextHash ^= SideKey;                         // 换手

            // 检查历史中出现的次数
            int count = _history.Count(s => s.Hash == nextHash);
            return count >= 2;
        }

        /// <summary>
        /// 标准走棋：更新棋盘、切换回合、记录历史并更新哈希
        /// </summary>
        public void Push(int from, int to)
        {
            sbyte piece = _cells[from];
            sbyte captured = _cells[to];

            // 1. 记录当前状态，包括当前的 LastMove 以便后续 Pop
            _history.Push(new GameState(from, to, captured, CurrentHash, LastMove));

            // 2. 【更新】最后一步移动
            LastMove = new Move(from, to);

            // 3. 增量更新哈希
            TogglePieceHash(from, piece);      // 移除起点棋子
            if (captured != 0)
                TogglePieceHash(to, captured); // 移除被吃掉的棋子
            TogglePieceHash(to, piece);        // 在终点放入棋子
            CurrentHash ^= SideKey;            // 切换回合哈希

            // 4. 物理移动
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

            // 恢复哈希和最后一步状态
            CurrentHash = last.Hash;
            LastMove = last.LastMoveBefore; // 【恢复】恢复到移动前的最后一步
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

        public IEnumerable<GameState> GetHistory() => _history;

        // --- 哈希逻辑 ---

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
            if (!IsRedTurn) // 习惯上黑方走棋时异或 SideKey
                CurrentHash ^= SideKey;
        }

        // --- 走法文本转换 ---

        public string GetChineseMoveName(int from, int to)
        {
            sbyte piece = _cells[from];
            if (piece == 0)
                return "";

            bool isRed = piece > 0;
            int fromR = from / 9, fromC = from % 9;
            int toR = to / 9, toC = to % 9;

            string name = GetPieceName(piece);
            int fromCol = isRed ? (9 - fromC) : (fromC + 1);
            int toCol = isRed ? (9 - toC) : (toC + 1);

            string action = (toR == fromR) ? "平" :
                            (isRed ? (toR < fromR ? "进" : "退") : (toR > fromR ? "进" : "退"));

            int targetValue;
            int type = Math.Abs(piece);
            if (type == 2 || type == 3 || type == 4) // 士、相、马
                targetValue = toCol;
            else
                targetValue = (action == "平") ? toCol : Math.Abs(toR - fromR);

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

        // --- 内部辅助方法 ---

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
            LastMove = null; // 加载新状态时清除最后一步
        }

        // 在 Board.cs 中添加
        public void RecordHistory(GameState state)
        {
            _history.Push(state);
            // 记录历史时同步更新 LastMove
            LastMove = new Move(state.From, state.To);
        }
    }
}