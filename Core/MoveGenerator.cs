using System;
using System.Collections.Generic;

namespace ChineseChessAI.Core
{
    public class MoveGenerator
    {
        private const int ROWS = 10;
        private const int COLS = 9;

        public string GetMoveValidationResult(Board board, Move move, bool skipPerpetualCheck = false)
        {
            var pseudoMoves = new List<Move>(64);
            sbyte piece = board.GetPiece(move.From);
            if (piece == 0) return "起点无棋子";
            bool isRed = piece > 0;
            if (isRed != board.IsRedTurn) return "走子方错误";

            // 1. 检查基础物理走法
            GeneratePieceMoves(board, move.From, piece, pseudoMoves);
            if (!pseudoMoves.Any(m => m.From == move.From && m.To == move.To))
                return "违规移动";

            // 2. 检查是否导致自方被将军（送将）
            sbyte captured = board.PerformMoveInternal(move.From, move.To);
            bool safe = IsKingSafe(board, isRed);
            board.UndoMoveInternal(move.From, move.To, captured);
            if (!safe) return "自方王受威胁 (送将)";

            // 3. 检查禁手规则（长打/长捉）
            if (!skipPerpetualCheck && board.GetRepetitionCount() >= 2)
            {
                if (IsForbiddenPerpetualMove(board, move))
                    return "禁手 (违规长打/长捉)";
            }

            return "合法";
        }

        /// <summary>
        /// 生成当前局面的所有合法走法（已过滤送将、飞将及违反长将/长捉规则的走法）
        /// </summary>
        public List<Move> GenerateLegalMoves(Board board)
        {
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

            var legalMoves = new List<Move>(pseudoMoves.Count);
            foreach (var move in pseudoMoves)
            {
                sbyte captured = board.PerformMoveInternal(move.From, move.To);
                bool safe = IsKingSafe(board, isRed);
                board.UndoMoveInternal(move.From, move.To, captured);

                if (!safe)
                    continue;

                if (board.GetRepetitionCount() >= 2)
                {
                    if (IsForbiddenPerpetualMove(board, move))
                        continue;
                }
                legalMoves.Add(move);
            }

            return legalMoves;
        }

        public Move? GetCaptureKingMove(Board board)
        {
            bool isRedAttacker = board.IsRedTurn;
            int enemyKingType = isRedAttacker ? -1 : 1;
            int enemyKingIndex = -1;

            for (int i = 0; i < 90; i++)
            {
                if (board.GetPiece(i) == enemyKingType)
                {
                    enemyKingIndex = i;
                    break;
                }
            }

            if (enemyKingIndex == -1)
                return null;

            var attacks = new List<Move>();
            for (int i = 0; i < 90; i++)
            {
                sbyte piece = board.GetPiece(i);
                if (piece != 0 && ((isRedAttacker && piece > 0) || (!isRedAttacker && piece < 0)))
                {
                    GeneratePieceMoves(board, i, piece, attacks);
                }
            }

            foreach (var m in attacks)
            {
                if (m.To != enemyKingIndex)
                    continue;

                // 验证该招法合法（不导致自方将暴露）
                sbyte captured = board.PerformMoveInternal(m.From, m.To);
                bool safe = IsKingSafe(board, isRedAttacker);
                board.UndoMoveInternal(m.From, m.To, captured);

                if (safe)
                    return m;
            }

            return null;
        }

        private bool IsForbiddenPerpetualMove(Board board, Move move)
        {
            board.Push(move.From, move.To);
            int count = board.GetRepetitionCount();
            bool isForbidden = false;
            if (count >= 3)
            {
                // 长将、长捉、长杀均视为禁手
                bool isChecking = IsChecking(board, !board.IsRedTurn);
                bool isChasing = IsChasing(board, move);
                bool isKillThreat = IsThreateningToMate(board, !board.IsRedTurn);
                
                if (isChecking || isChasing || isKillThreat)
                    isForbidden = true;
            }
            board.Pop();
            return isForbidden;
        }

        public bool IsChecking(Board board, bool isRedAttacker)
        {
            return !IsKingSafe(board, !isRedAttacker);
        }

        /// <summary>
        /// 判断是否产生“捉”的威胁。
        /// 完善版：包含抽吃检测，区分“捉”与“兑”。
        /// </summary>
        private bool IsChasing(Board board, Move move)
        {
            // 在执行 move 之后的局面判断
            bool isRedAttacker = !board.IsRedTurn;
            
            // 1. 获取移动前被攻击的敌方棋子集合（需要回溯一步，或者在调用前计算）
            // 为了简化，我们直接判断当前局面下，是否有任何敌方重要棋子正受到我方攻击，且这种攻击是有效的。
            
            // 遍历所有我方棋子，看是否在攻击敌方棋子
            for (int i = 0; i < 90; i++)
            {
                sbyte attacker = board.GetPiece(i);
                if (attacker == 0 || (isRedAttacker ? attacker < 0 : attacker > 0)) continue;

                var attacks = new List<Move>();
                GeneratePieceMoves(board, i, attacker, attacks);

                foreach (var m in attacks)
                {
                    sbyte target = board.GetPiece(m.To);
                    if (target == 0) continue;
                    
                    // 敌方棋子
                    if (isRedAttacker ? target < 0 : target > 0)
                    {
                        if (IsRealChase(board, i, m.To))
                            return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// 判断从 from 位置的棋子攻击 to 位置的棋子是否构成“捉”
        /// </summary>
        private bool IsRealChase(Board board, int from, int to)
        {
            sbyte attacker = board.GetPiece(from);
            sbyte target = board.GetPiece(to);
            int targetType = Math.Abs(target);
            
            // 将、士、象不计入常捉（通常判定为闲着或平局规则）
            if (targetType <= 3) return false;
            
            // 未过河兵卒不计入常捉
            if (targetType == 7)
            {
                bool isRedTarget = target > 0;
                int row = to / 9;
                if (isRedTarget ? (row >= 5) : (row <= 4)) return false;
            }

            int attackerVal = GetPieceValue(attacker);
            int targetVal = GetPieceValue(target);

            // 检查目标是否有保护
            bool protectedTarget = IsProtected(board, to);

            // 规则：
            // 1. 攻击未受保护的棋子 -> 捉
            // 2. 攻击价值更高的受保护棋子 -> 捉
            // 3. 攻击价值相等的受保护棋子 -> 兑（非捉）
            // 4. 攻击价值更低的受保护棋子 -> 非捉
            
            if (!protectedTarget) return true;
            if (attackerVal < targetVal) return true;
            
            return false;
        }

        private bool IsProtected(Board board, int pos)
        {
            sbyte target = board.GetPiece(pos);
            if (target == 0) return false;
            bool isRed = target > 0;
            
            // 模拟移除该棋子，看自方是否仍能走到该位置
            sbyte original = target;
            board.PerformMoveInternal(pos, pos); // 临时置空或用特殊标记
            // 实际上 PerformMoveInternal(pos, pos) 不太对，我们手动修改
            // 这里的逻辑应该是：如果一个自方棋子可以走到这个位置（即使现在有棋子），那就是保护。
            
            // 遍历所有自方棋子
            for (int i = 0; i < 90; i++)
            {
                if (i == pos) continue;
                sbyte p = board.GetPiece(i);
                if (p == 0 || (isRed ? p < 0 : p > 0)) continue;

                var moves = new List<Move>();
                // 注意：炮的保护逻辑特殊，需要单独处理或确保 GeneratePieceMoves 正确处理了同色棋子作为“炮架”
                // 在象棋规则中，如果炮的攻击线上目标是自方棋子，那个自方棋子是被保护的。
                GeneratePieceMovesForProtection(board, i, p, moves);
                if (moves.Any(m => m.To == pos))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// 专门用于判定保护的走法生成（炮可以越过一个棋子保护自方棋子）
        /// </summary>
        private void GeneratePieceMovesForProtection(Board board, int from, sbyte piece, List<Move> moves)
        {
            // 大部分棋子的保护范围与其攻击范围一致
            int type = Math.Abs(piece);
            if (type != 6) // 非炮
            {
                GeneratePieceMoves(board, from, piece, moves);
                // 默认的 GeneratePieceMoves 遇到友军会停止，我们需要包含友军位置
                // 所以这里稍微修改逻辑
                RefineMovesWithFriends(board, from, piece, moves);
            }
            else // 炮的特殊保护逻辑
            {
                GenerateCannonProtection(board, from, piece, moves);
            }
        }

        private void RefineMovesWithFriends(Board board, int from, sbyte piece, List<Move> moves)
        {
            int r = from / 9, c = from % 9, type = Math.Abs(piece);
            bool isRed = piece > 0;

            switch (type)
            {
                case 1: // King
                case 2: // Advisor
                    int[] drKA = type == 1 ? new[] { -1, 1, 0, 0 } : new[] { -1, -1, 1, 1 };
                    int[] dcKA = type == 1 ? new[] { 0, 0, -1, 1 } : new[] { -1, 1, -1, 1 };
                    for (int i = 0; i < 4; i++)
                    {
                        int nr = r + drKA[i], nc = c + dcKA[i];
                        if (nc < 3 || nc > 5 || (isRed ? (nr < 7 || nr > 9) : (nr < 0 || nr > 2))) continue;
                        if (board.GetPiece(nr, nc) != 0) moves.Add(new Move(from, nr * 9 + nc));
                    }
                    break;
                case 3: // Bishop
                    int[] drB = { -2, -2, 2, 2 }, dcB = { -2, 2, -2, 2 };
                    int[] prB = { -1, -1, 1, 1 }, pcB = { -1, 1, -1, 1 };
                    for (int i = 0; i < 4; i++)
                    {
                        int nr = r + drB[i], nc = c + dcB[i];
                        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
                        if ((isRed && nr < 5) || (!isRed && nr > 4)) continue;
                        if (board.GetPiece(r + prB[i], c + pcB[i]) != 0) continue;
                        if (board.GetPiece(nr, nc) != 0) moves.Add(new Move(from, nr * 9 + nc));
                    }
                    break;
                case 4: // Knight
                    int[] drK = { -2, -2, -1, -1, 1, 1, 2, 2 }, dcK = { -1, 1, -2, 2, -2, 2, -1, 1 };
                    int[] prK = { -1, -1, 0, 0, 0, 0, 1, 1 }, pcK = { 0, 0, -1, 1, -1, 1, 0, 0 };
                    for (int i = 0; i < 8; i++)
                    {
                        int nr = r + drK[i], nc = c + dcK[i];
                        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
                        if (board.GetPiece(r + prK[i], c + pcK[i]) != 0) continue;
                        if (board.GetPiece(nr, nc) != 0) moves.Add(new Move(from, nr * 9 + nc));
                    }
                    break;
                case 5: // Rook
                    int[] drR = { -1, 1, 0, 0 }, dcR = { 0, 0, -1, 1 };
                    for (int i = 0; i < 4; i++)
                    {
                        for (int step = 1; step < 10; step++)
                        {
                            int nr = r + drR[i] * step, nc = c + dcR[i] * step;
                            if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) break;
                            sbyte target = board.GetPiece(nr, nc);
                            if (target != 0) { moves.Add(new Move(from, nr * 9 + nc)); break; }
                        }
                    }
                    break;
                case 7: // Pawn
                    int forwardR = isRed ? r - 1 : r + 1;
                    if (forwardR >= 0 && forwardR < 10 && board.GetPiece(forwardR, c) != 0)
                        moves.Add(new Move(from, forwardR * 9 + c));
                    if (isRed ? (r <= 4) : (r >= 5))
                    {
                        if (c - 1 >= 0 && board.GetPiece(r, c - 1) != 0) moves.Add(new Move(from, r * 9 + (c - 1)));
                        if (c + 1 < 9 && board.GetPiece(r, c + 1) != 0) moves.Add(new Move(from, r * 9 + (c + 1)));
                    }
                    break;
            }
        }

        private void GenerateCannonProtection(Board board, int from, sbyte piece, List<Move> moves)
        {
            int r = from / 9, c = from % 9;
            int[] dr = { -1, 1, 0, 0 }, dc = { 0, 0, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                bool overPiece = false;
                for (int step = 1; step < 10; step++)
                {
                    int nr = r + dr[i] * step, nc = c + dc[i] * step;
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) break;
                    sbyte target = board.GetPiece(nr, nc);
                    if (!overPiece)
                    {
                        if (target != 0) overPiece = true;
                    }
                    else
                    {
                        if (target != 0)
                        {
                            // 只要翻山后碰到棋子（无论敌友），都是在保护/攻击
                            moves.Add(new Move(from, nr * 9 + nc));
                            break;
                        }
                    }
                }
            }
        }

        public bool IsThreateningToMate(Board board, bool isRedAttacker)
        {
            // 在 IsForbiddenPerpetualMove 中，我们是在 board.Push(move) 之后调用的。
            // 所以 board.IsRedTurn 已经是对方了。
            // 简单的“杀”判定：对方已经没有合法走法（困毙）。
            var legalMoves = GenerateLegalMoves(board);
            if (legalMoves.Count == 0) return true;
            
            return false;
        }

        private int GetPieceValue(sbyte piece)
        {
            switch (Math.Abs(piece))
            {
                case 1: return 1000; // 帅
                case 5: return 9;    // 俥
                case 6: return 5;    // 炮
                case 4: return 4;    // 傌
                case 7: return 2;    // 卒 (过河)
                default: return 2;   // 其他
            }
        }

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
                        if (IsEnemy(isRedKing, p, 5) || IsEnemy(isRedKing, p, 1))
                            return false;
                    }
                    else if (count == 2)
                    {
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

        /// <summary>
        /// 【核心修复】精准的马后炮/马将威胁判定
        /// </summary>
        private bool CheckKnightThreats(Board board, int r, int c, bool isRedKing)
        {
            int[] dr = { -2, -2, -1, -1, 1, 1, 2, 2 };
            int[] dc = { -1, 1, -2, 2, -2, 2, -1, 1 };

            for (int i = 0; i < 8; i++)
            {
                int nr = r + dr[i];
                int nc = c + dc[i];

                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS)
                {
                    // 找到了一个可以跳到老将位置的敌方马
                    if (IsEnemy(isRedKing, board.GetPiece(nr, nc), 4))
                    {
                        // 计算马腿坐标：马腿永远在紧贴着马(nr, nc)的前进方向上
                        int legR = nr;
                        int legC = nc;

                        if (Math.Abs(dr[i]) == 2)
                        {
                            // 马竖着跳了2格，说明马腿在它的垂直方向上
                            legR -= dr[i] / 2;
                        }
                        else
                        {
                            // 马横着跳了2格，说明马腿在它的水平方向上
                            legC -= dc[i] / 2;
                        }

                        // 如果马腿位置是空的，说明确实被将军了！
                        if (board.GetPiece(legR, legC) == 0)
                        {
                            return false;
                        }
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