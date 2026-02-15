namespace ChineseChessAI.Core
{
    public class MoveGenerator
    {
        private const int ROWS = 10;
        private const int COLS = 9;

        /// <summary>
        /// 生成当前局面的所有合法走法（已过滤送将、飞将及违反长将/长捉规则的走法）
        /// </summary>
        public List<Move> GenerateLegalMoves(Board board)
        {
            // 1. 生成所有伪合法走法 (Pseudo-Legal)
            var pseudoMoves = new List<Move>(64);
            bool isRed = board.IsRedTurn;

            for (int i = 0; i < 90; i++)
            {
                sbyte piece = board.GetPiece(i);
                if (piece == 0)
                    continue;

                if ((isRed && piece > 0) || (!isRed && piece < 0))
                {
                    GeneratePieceMoves(board, i, piece, pseudoMoves);
                }
            }

            // 2. 过滤非法走法
            var legalMoves = new List<Move>(pseudoMoves.Count);
            foreach (var move in pseudoMoves)
            {
                // A. 试走并检查己方帅位安全（包括飞将检测）
                sbyte captured = board.PerformMoveInternal(move.From, move.To);
                bool safe = IsKingSafe(board, isRed);
                board.UndoMoveInternal(move.From, move.To, captured);

                if (!safe)
                    continue;

                // B. 长将与长捉检测 (重复局面判定)
                // 如果当前步会导致第3次及以上重复，则检查其攻击属性
                if (board.GetRepetitionCount() >= 2) // 如果当前哈希在历史中已出现2次，下一步就是第3次
                {
                    if (IsForbiddenPerpetualMove(board, move))
                        continue;
                }

                legalMoves.Add(move);
            }

            return legalMoves;
        }

        /// <summary>
        /// 判定某一着棋是否属于“禁着”（长将或禁止的长捉）
        /// </summary>
        private bool IsForbiddenPerpetualMove(Board board, Move move)
        {
            // 模拟推入局面
            board.Push(move.From, move.To);
            int count = board.GetRepetitionCount();

            bool isForbidden = false;
            if (count >= 3)
            {
                // 分析历史循环中的步法属性
                // 按照中国象棋规则：单方长将必判负，长捉大子通常也禁止
                bool isChecking = IsChecking(board, !board.IsRedTurn); // 刚走完的那一方是否在将军
                bool isChasing = IsChasing(board, move);             // 刚走完的那一着是否在“捉”

                // 在 AI 层面，为了简化逻辑并保持严谨性：
                // 如果连续 3 次重复局面且每一步都是攻击性动作（将或捉），则视为禁着
                if (isChecking || isChasing)
                {
                    isForbidden = true;
                }
            }

            board.Pop();
            return isForbidden;
        }

        /// <summary>
        /// 检查当前回合方是否正在将军
        /// </summary>
        public bool IsChecking(Board board, bool isRedAttacker)
        {
            // 对方老将如果不安全，说明我方正在将军
            return !IsKingSafe(board, !isRedAttacker);
        }

        /// <summary>
        /// 判定当前走法是否属于“捉”（针对车、马、炮等大子的攻击）
        /// </summary>
        private bool IsChasing(Board board, Move move)
        {
            sbyte attacker = board.GetPiece(move.To);
            bool isRedAttacker = attacker > 0;

            // 1. 生成移动后该棋子的所有走法
            var attacks = new List<Move>();
            GeneratePieceMoves(board, move.To, attacker, attacks);

            foreach (var m in attacks)
            {
                sbyte target = board.GetPiece(m.To);
                if (target == 0)
                    continue;

                // 2. 如果目标是对方的大子（车、马、炮）
                int targetType = Math.Abs(target);
                if ((isRedAttacker && target < 0) || (!isRedAttacker && target > 0))
                {
                    if (targetType >= 4 && targetType <= 6) // 4:马, 5:车, 6:炮
                    {
                        // 3. 简单的“捉”判定：攻击了大子且对方该子不在保护下（或属于不计代价的捉）
                        // 在复杂规则中需判定“保护”，此处实现为：只要产生了新的大子威胁即视为攻击
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// 检查指定颜色的老将是否安全（未被将军，且无飞将）
        /// </summary>
        public bool IsKingSafe(Board board, bool checkRed)
        {
            int kingIndex = -1;
            int kingPiece = checkRed ? 1 : -1;

            for (int i = 0; i < 90; i++)
            {
                if (board.GetPiece(i) == kingPiece)
                {
                    kingIndex = i;
                    break;
                }
            }
            if (kingIndex == -1)
                return false;

            int kr = kingIndex / 9, kc = kingIndex % 9;

            return CheckLinearThreats(board, kr, kc, checkRed) &&
                   CheckKnightThreats(board, kr, kc, checkRed) &&
                   CheckPawnThreats(board, kr, kc, checkRed);
        }

        private bool CheckLinearThreats(Board board, int r, int c, bool isRedKing)
        {
            int[] dr = { -1, 1, 0, 0 }, dc = { 0, 0, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int count = 0;
                for (int step = 1; step < 10; step++)
                {
                    int nr = r + dr[i] * step, nc = c + dc[i] * step;
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                        break;
                    sbyte p = board.GetPiece(nr, nc);
                    if (p == 0)
                        continue;
                    count++;
                    if (count == 1)
                    {
                        // 敌方车或老将（飞将检测：类型1）
                        if (IsEnemy(isRedKing, p, 5) || IsEnemy(isRedKing, p, 1))
                            return false;
                    }
                    else if (count == 2)
                    {
                        // 敌方炮
                        if (IsEnemy(isRedKing, p, 6))
                            return false;
                        break;
                    }
                    else
                        break;
                }
            }
            return true;
        }

        private bool CheckKnightThreats(Board board, int r, int c, bool isRedKing)
        {
            int[] dr = { -2, -2, -1, -1, 1, 1, 2, 2 }, dc = { -1, 1, -2, 2, -2, 2, -1, 1 };
            int[] lr = { -1, -1, 0, 0, 0, 0, 1, 1 }, lc = { 0, 0, -1, 1, -1, 1, 0, 0 };
            for (int i = 0; i < 8; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS)
                {
                    if (IsEnemy(isRedKing, board.GetPiece(nr, nc), 4))
                    {
                        if (board.GetPiece(r + lr[i], c + lc[i]) == 0)
                            return false;
                    }
                }
            }
            return true;
        }

        private bool CheckPawnThreats(Board board, int r, int c, bool isRedKing)
        {
            if (isRedKing)
            {
                if (r - 1 >= 0 && IsEnemy(isRedKing, board.GetPiece(r - 1, c), 7))
                    return false;
                if (c - 1 >= 0 && IsEnemy(isRedKing, board.GetPiece(r, c - 1), 7))
                    return false;
                if (c + 1 < COLS && IsEnemy(isRedKing, board.GetPiece(r, c + 1), 7))
                    return false;
            }
            else
            {
                if (r + 1 < ROWS && IsEnemy(isRedKing, board.GetPiece(r + 1, c), 7))
                    return false;
                if (c - 1 >= 0 && IsEnemy(isRedKing, board.GetPiece(r, c - 1), 7))
                    return false;
                if (c + 1 < COLS && IsEnemy(isRedKing, board.GetPiece(r, c + 1), 7))
                    return false;
            }
            return true;
        }

        private bool IsEnemy(bool isRedSelf, sbyte p, int type) =>
            p != 0 && Math.Abs(p) == type && (isRedSelf ? p < 0 : p > 0);

        // --- 物理走法生成 (Pseudo-Legal) ---

        private void GeneratePieceMoves(Board board, int from, sbyte piece, List<Move> moves)
        {
            int r = from / 9, c = from % 9, type = Math.Abs(piece);
            bool isRed = piece > 0;
            switch (type)
            {
                case 1:
                    GenerateKingMoves(board, from, r, c, moves, isRed);
                    break;
                case 2:
                    GenerateAdvisorMoves(board, from, r, c, moves, isRed);
                    break;
                case 3:
                    GenerateBishopMoves(board, from, r, c, moves, isRed);
                    break;
                case 4:
                    GenerateKnightMoves(board, from, r, c, moves);
                    break;
                case 5:
                    GenerateLinearMoves(board, from, r, c, moves, false);
                    break;
                case 6:
                    GenerateLinearMoves(board, from, r, c, moves, true);
                    break;
                case 7:
                    GeneratePawnMoves(board, from, r, c, moves, isRed);
                    break;
            }
        }

        private void GenerateKingMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int[] dr = { -1, 1, 0, 0 }, dc = { 0, 0, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nc < 3 || nc > 5 || (isRed ? (nr < 7 || nr > 9) : (nr < 0 || nr > 2)))
                    continue;
                TryAddMove(board, from, nr, nc, moves);
            }
        }

        private void GenerateAdvisorMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int[] dr = { -1, -1, 1, 1 }, dc = { -1, 1, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nc < 3 || nc > 5 || (isRed ? (nr < 7 || nr > 9) : (nr < 0 || nr > 2)))
                    continue;
                TryAddMove(board, from, nr, nc, moves);
            }
        }

        private void GenerateBishopMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int[] dr = { -2, -2, 2, 2 }, dc = { -2, 2, -2, 2 };
            int[] pr = { -1, -1, 1, 1 }, pc = { -1, 1, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                    continue;
                if ((isRed && nr < 5) || (!isRed && nr > 4))
                    continue;
                if (board.GetPiece(r + pr[i], c + pc[i]) != 0)
                    continue;
                TryAddMove(board, from, nr, nc, moves);
            }
        }

        private void GenerateKnightMoves(Board board, int from, int r, int c, List<Move> moves)
        {
            int[] dr = { -2, -2, -1, -1, 1, 1, 2, 2 }, dc = { -1, 1, -2, 2, -2, 2, -1, 1 };
            int[] pr = { -1, -1, 0, 0, 0, 0, 1, 1 }, pc = { 0, 0, -1, 1, -1, 1, 0, 0 };
            for (int i = 0; i < 8; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                    continue;
                if (board.GetPiece(r + pr[i], c + pc[i]) != 0)
                    continue;
                TryAddMove(board, from, nr, nc, moves);
            }
        }

        private void GenerateLinearMoves(Board board, int from, int r, int c, List<Move> moves, bool isCannon)
        {
            int[] dr = { -1, 1, 0, 0 }, dc = { 0, 0, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                bool overPiece = false;
                for (int step = 1; step < 10; step++)
                {
                    int nr = r + dr[i] * step, nc = c + dc[i] * step;
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                        break;
                    sbyte target = board.GetPiece(nr, nc);
                    if (!isCannon)
                    {
                        if (target == 0)
                            moves.Add(new Move(from, nr * 9 + nc));
                        else
                        {
                            if (IsEnemySimple(board.GetPiece(from), target))
                                moves.Add(new Move(from, nr * 9 + nc));
                            break;
                        }
                    }
                    else
                    {
                        if (!overPiece)
                        {
                            if (target == 0)
                                moves.Add(new Move(from, nr * 9 + nc));
                            else
                                overPiece = true;
                        }
                        else if (target != 0)
                        {
                            if (IsEnemySimple(board.GetPiece(from), target))
                                moves.Add(new Move(from, nr * 9 + nc));
                            break;
                        }
                    }
                }
            }
        }

        private void GeneratePawnMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int forwardR = isRed ? r - 1 : r + 1;
            if (forwardR >= 0 && forwardR < 10)
                TryAddMove(board, from, forwardR, c, moves);
            if (isRed ? (r <= 4) : (r >= 5))
            {
                if (c - 1 >= 0)
                    TryAddMove(board, from, r, c - 1, moves);
                if (c + 1 < 9)
                    TryAddMove(board, from, r, c + 1, moves);
            }
        }

        private void TryAddMove(Board board, int from, int nr, int nc, List<Move> moves)
        {
            sbyte target = board.GetPiece(nr, nc);
            if (target == 0 || IsEnemySimple(board.GetPiece(from), target))
                moves.Add(new Move(from, nr * 9 + nc));
        }

        private bool IsEnemySimple(sbyte p1, sbyte p2) => (p1 > 0 && p2 < 0) || (p1 < 0 && p2 > 0);
    }
}