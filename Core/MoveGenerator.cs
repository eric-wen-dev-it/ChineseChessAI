using System;
using System.Collections.Generic;

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
                if (board.GetRepetitionCount() >= 2)
                {
                    if (IsForbiddenPerpetualMove(board, move))
                        continue;
                }

                legalMoves.Add(move);
            }

            return legalMoves;
        }

        /// <summary>
        /// 【核心新增】判定某一动作是否能直接“击杀”对方老将。
        /// 用于实现 AI 的强制击杀逻辑。
        /// </summary>
        public bool CanCaptureKing(Board board, Move move)
        {
            bool isRedAttacker = board.IsRedTurn;

            // 1. 模拟走棋
            sbyte captured = board.PerformMoveInternal(move.From, move.To);

            // 2. 检查对方老将是否在当前局面下处于被攻击状态（即老将不安全）
            // 在象棋规则逻辑中，如果对方老将“不安全”，意味着这一步可以直接吃将
            bool canCapture = !IsKingSafe(board, !isRedAttacker);

            // 3. 撤销模拟
            board.UndoMoveInternal(move.From, move.To, captured);

            return canCapture;
        }

        private bool IsForbiddenPerpetualMove(Board board, Move move)
        {
            board.Push(move.From, move.To);
            int count = board.GetRepetitionCount();

            bool isForbidden = false;
            if (count >= 3)
            {
                bool isChecking = IsChecking(board, !board.IsRedTurn);
                bool isChasing = IsChasing(board, move);

                if (isChecking || isChasing)
                {
                    isForbidden = true;
                }
            }

            board.Pop();
            return isForbidden;
        }

        public bool IsChecking(Board board, bool isRedAttacker)
        {
            return !IsKingSafe(board, !isRedAttacker);
        }

        private bool IsChasing(Board board, Move move)
        {
            sbyte attacker = board.GetPiece(move.To);
            if (attacker == 0)
                return false;
            bool isRedAttacker = attacker > 0;

            var attacks = new List<Move>();
            GeneratePieceMoves(board, move.To, attacker, attacks);

            foreach (var m in attacks)
            {
                sbyte target = board.GetPiece(m.To);
                if (target == 0)
                    continue;

                int targetType = Math.Abs(target);
                if ((isRedAttacker && target < 0) || (!isRedAttacker && target > 0))
                {
                    if (targetType >= 4 && targetType <= 6) // 4:马, 5:车, 6:炮
                    {
                        return true;
                    }
                }
            }
            return false;
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

            // 如果棋盘上已经没有将（被吃了），直接返回不安全
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