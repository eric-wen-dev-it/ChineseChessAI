namespace ChineseChessAI.Core
{
    /// <summary>
    /// 存储每一步的历史状态，用于撤销和长捉/长将检测
    /// </summary>
    public record GameState(int From, int To, sbyte Captured, ulong Hash, Move? LastMoveBefore, Dictionary<ulong, int>? HashCountsSnapshot = null);

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

        public bool LastMoveWasIrreversible { get; private set; } = false;

        // 【核心修复 CE-3】：增量维护子力分数，并提供公开访问
        public float RedMaterial { get; private set; } = 0;
        public float BlackMaterial { get; private set; } = 0;

        public Board()
        {
            Reset();
        }

        public void Reset()
        {
            Array.Clear(_cells, 0, _cells.Length);
            RedMaterial = 0;
            BlackMaterial = 0;

            // --- 摆放黑方 (第 0-3 行) ---
            SetPieceWithMaterial(0, -5);
            SetPieceWithMaterial(8, -5); // 黑车
            SetPieceWithMaterial(1, -4);
            SetPieceWithMaterial(7, -4); // 黑马
            SetPieceWithMaterial(2, -3);
            SetPieceWithMaterial(6, -3); // 黑象
            SetPieceWithMaterial(3, -2);
            SetPieceWithMaterial(5, -2); // 黑士
            SetPieceWithMaterial(4, -1);             // 黑将
            SetPieceWithMaterial(19, -6);
            SetPieceWithMaterial(25, -6); // 黑炮
            for (int i = 0; i < 9; i += 2)
                SetPieceWithMaterial(27 + i, -7); // 黑卒

            // --- 摆放红方 (第 6-9 行) ---
            for (int i = 0; i < 9; i += 2)
                SetPieceWithMaterial(6 * 9 + i, 7);  // 红兵
            SetPieceWithMaterial(7 * 9 + 1, 6);
            SetPieceWithMaterial(7 * 9 + 7, 6); // 红炮
            SetPieceWithMaterial(9 * 9 + 0, 5);
            SetPieceWithMaterial(9 * 9 + 8, 5); // 红车
            SetPieceWithMaterial(9 * 9 + 1, 4);
            SetPieceWithMaterial(9 * 9 + 7, 4); // 红马
            SetPieceWithMaterial(9 * 9 + 2, 3);
            SetPieceWithMaterial(9 * 9 + 6, 3); // 红相
            SetPieceWithMaterial(9 * 9 + 3, 2);
            SetPieceWithMaterial(9 * 9 + 5, 2); // 红仕
            SetPieceWithMaterial(9 * 9 + 4, 1);                     // 红帅

            IsRedTurn = true;
            _history.Clear();
            _hashCounts.Clear(); // 清空计数表
            LastMove = null;
            LastMoveWasIrreversible = false;

            CalculateFullHash();
            _hashCounts[CurrentHash] = 1; // 初始局面计数为 1
        }

        private void SetPieceWithMaterial(int pos, sbyte p)
        {
            _cells[pos] = p;
            UpdateMaterial(p, 1); // 增加子力
        }

        private void UpdateMaterial(sbyte p, int sign)
        {
            if (p == 0)
                return;
            float val = Math.Abs(p) switch
            {
                1 => 0,
                2 => 2,
                3 => 2,
                4 => 4,
                5 => 9,
                6 => 4.5f,
                7 => 1,
                _ => 0
            };
            if (p > 0)
                RedMaterial += val * sign;
            else
                BlackMaterial += val * sign;
        }

        public Board Clone()
        {
            var newBoard = new Board();
            // 【BUG C 修复】：new Board() 已经写入了初始哈希计数，必须清空后再复制
            newBoard._hashCounts.Clear();

            Array.Copy(this._cells, newBoard._cells, 90);
            newBoard.IsRedTurn = this.IsRedTurn;
            newBoard.CurrentHash = this.CurrentHash;
            newBoard.LastMove = this.LastMove;
            newBoard.LastMoveWasIrreversible = this.LastMoveWasIrreversible;
            newBoard.RedMaterial = this.RedMaterial;
            newBoard.BlackMaterial = this.BlackMaterial;

            // 深度克隆历史栈
            var historyArray = this._history.ToArray();
            Array.Reverse(historyArray);
            foreach (var state in historyArray)
                newBoard._history.Push(state);

            // 深度克隆哈希计数表
            foreach (var kvp in this._hashCounts)
                newBoard._hashCounts[kvp.Key] = kvp.Value;

            return newBoard;
        }

        public sbyte GetPiece(int row, int col) => _cells[row * 9 + col];
        public sbyte GetPiece(int index) => _cells[index];

        public bool WillCauseThreefoldRepetition(int from, int to)
        {
            sbyte piece = _cells[from];
            sbyte captured = _cells[to];
            ulong nextHash = CurrentHash;

            nextHash ^= Zobrist.GetPieceKey(from, piece);
            if (captured != 0)
                nextHash ^= Zobrist.GetPieceKey(to, captured);
            nextHash ^= Zobrist.GetPieceKey(to, piece);
            nextHash ^= Zobrist.SideKey;

            // 【核心提速修复】：O(1) 字典瞬间查询，替换原有的 O(n) LINQ 遍历
            _hashCounts.TryGetValue(nextHash, out int count);
            return count >= 2;
        }

        public void Push(int from, int to)
        {
            sbyte piece = _cells[from];
            sbyte captured = _cells[to];

            // 【BUG E 修复】：任何兵卒走动（包括横移）均视为不可逆
            bool isPawnMove = Math.Abs(piece) == 7;

            LastMoveWasIrreversible = (captured != 0) || isPawnMove;

            Dictionary<ulong, int>? snapshot = null;
            if (LastMoveWasIrreversible)
            {
                snapshot = new Dictionary<ulong, int>(_hashCounts);
            }

            _history.Push(new GameState(from, to, captured, CurrentHash, LastMove, snapshot));
            LastMove = new Move(from, to);

            TogglePieceHash(from, piece);
            if (captured != 0)
            {
                TogglePieceHash(to, captured);
                UpdateMaterial(captured, -1); // 【新增】：吃子时减少对方子力
            }
            TogglePieceHash(to, piece);
            CurrentHash ^= Zobrist.SideKey;

            _cells[to] = piece;
            _cells[from] = 0;
            IsRedTurn = !IsRedTurn;

            // 【BUG 3 修复】：如果是不可逆招法，清空哈希历史计数
            if (LastMoveWasIrreversible)
            {
                _hashCounts.Clear();
            }

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

            if (last.HashCountsSnapshot != null)
            {
                _hashCounts.Clear();
                foreach (var kvp in last.HashCountsSnapshot)
                    _hashCounts[kvp.Key] = kvp.Value;
            }

            // 【新增】：恢复被吃掉的子力
            if (last.Captured != 0)
            {
                UpdateMaterial(last.Captured, 1);
            }

            _cells[last.From] = _cells[last.To];
            _cells[last.To] = last.Captured;
            IsRedTurn = !IsRedTurn;
            CurrentHash = last.Hash; // 恢复旧局面的 Hash
            LastMove = last.LastMoveBefore;
            LastMoveWasIrreversible = false; // Pop 后无法简单反推，设为 false
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
            CurrentHash ^= Zobrist.GetPieceKey(pos, piece);
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
                CurrentHash ^= Zobrist.SideKey;
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

            // 【核心修复 BUG-3】：重新计算材料分数
            RedMaterial = 0;
            BlackMaterial = 0;
            for (int i = 0; i < 90; i++)
                UpdateMaterial(_cells[i], 1);

            CalculateFullHash();

            _history.Clear();
            _hashCounts.Clear();
            _hashCounts[CurrentHash] = 1; // 载入新局面时初始化计数
            LastMove = null;
        }

    }
}