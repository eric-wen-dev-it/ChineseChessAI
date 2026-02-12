using System;
using System.Collections.Generic;

namespace ChineseChessAI.Core
{
    public class MoveGenerator
    {
        private const int ROWS = 10;
        private const int COLS = 9;

        /// <summary>
        /// 生成当前局面的所有物理合法走法
        /// </summary>
        public List<Move> GenerateLegalMoves(Board board)
        {
            var moves = new List<Move>();
            bool isRed = board.IsRedTurn;

            for (int i = 0; i < 90; i++)
            {
                sbyte piece = board.GetPiece(i / 9, i % 9);
                if (piece == 0)
                    continue;

                // 只处理当前回合方的棋子
                if ((isRed && piece > 0) || (!isRed && piece < 0))
                {
                    GeneratePieceMoves(board, i, piece, moves);
                }
            }
            return moves;
        }

        private void GeneratePieceMoves(Board board, int from, sbyte piece, List<Move> moves)
        {
            int r = from / 9;
            int c = from % 9;
            int type = Math.Abs(piece);
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

        // 1. 帅/将逻辑：限制在九宫格
        private void GenerateKingMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int[] dr = { -1, 1, 0, 0 };
            int[] dc = { 0, 0, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nc < 3 || nc > 5)
                    continue;
                if (isRed && (nr < 7 || nr > 9))
                    continue;
                if (!isRed && (nr < 0 || nr > 2))
                    continue;

                TryAddMove(board, from, nr, nc, moves);
            }
        }

        // 2. 仕/士逻辑：斜行且限制在九宫格
        private void GenerateAdvisorMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int[] dr = { -1, -1, 1, 1 };
            int[] dc = { -1, 1, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nc < 3 || nc > 5)
                    continue;
                if (isRed && (nr < 7 || nr > 9))
                    continue;
                if (!isRed && (nr < 0 || nr > 2))
                    continue;

                TryAddMove(board, from, nr, nc, moves);
            }
        }

        // 3. 相/象逻辑：田字格且不能过河，需检查塞象眼
        private void GenerateBishopMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int[] dr = { -2, -2, 2, 2 };
            int[] dc = { -2, 2, -2, 2 };
            int[] pr = { -1, -1, 1, 1 }; // 象眼位置
            int[] pc = { -1, 1, -1, 1 };

            for (int i = 0; i < 4; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                    continue;
                if (isRed && nr < 5)
                    continue;
                if (!isRed && nr > 4)
                    continue;

                if (board.GetPiece(r + pr[i], c + pc[i]) != 0)
                    continue; // 塞象眼
                TryAddMove(board, from, nr, nc, moves);
            }
        }

        // 4. 马逻辑：日字格，需检查蹩马腿
        private void GenerateKnightMoves(Board board, int from, int r, int c, List<Move> moves)
        {
            int[] dr = { -2, -2, -1, -1, 1, 1, 2, 2 };
            int[] dc = { -1, 1, -2, 2, -2, 2, -1, 1 };
            int[] pr = { -1, -1, 0, 0, 0, 0, 1, 1 }; // 马腿位置
            int[] pc = { 0, 0, -1, 1, -1, 1, 0, 0 };

            for (int i = 0; i < 8; i++)
            {
                int nr = r + dr[i], nc = c + dc[i];
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                    continue;
                if (board.GetPiece(r + pr[i], c + pc[i]) != 0)
                    continue; // 蹩马腿

                TryAddMove(board, from, nr, nc, moves);
            }
        }

        // 5 & 6. 车/炮逻辑：直线扫描
        private void GenerateLinearMoves(Board board, int from, int r, int c, List<Move> moves, bool isCannon)
        {
            int[] dr = { -1, 1, 0, 0 };
            int[] dc = { 0, 0, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                bool overPiece = false;
                for (int step = 1; step < 10; step++)
                {
                    int nr = r + dr[i] * step, nc = c + dc[i] * step;
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                        break;

                    sbyte target = board.GetPiece(nr, nc);
                    if (!isCannon) // 车的逻辑
                    {
                        if (target == 0)
                            moves.Add(new Move(from, nr * 9 + nc));
                        else
                        {
                            if (IsEnemy(board.GetPiece(r, c), target))
                                moves.Add(new Move(from, nr * 9 + nc));
                            break;
                        }
                    }
                    else // 炮的逻辑
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
                            if (IsEnemy(board.GetPiece(r, c), target))
                                moves.Add(new Move(from, nr * 9 + nc));
                            break;
                        }
                    }
                }
            }
        }

        // 7. 兵/卒逻辑：过河前只能前行，过河后可横行
        private void GeneratePawnMoves(Board board, int from, int r, int c, List<Move> moves, bool isRed)
        {
            int forwardR = isRed ? r - 1 : r + 1;
            if (forwardR >= 0 && forwardR < 10)
                TryAddMove(board, from, forwardR, c, moves);

            bool crossed = isRed ? (r <= 4) : (r >= 5);
            if (crossed)
            {
                if (c - 1 >= 0)
                    TryAddMove(board, from, r, c - 1, moves);
                if (c + 1 < 9)
                    TryAddMove(board, from, r, c + 1, moves);
            }
        }

        // 辅助方法：检查目标位置并添加走法
        private void TryAddMove(Board board, int from, int nr, int nc, List<Move> moves)
        {
            sbyte target = board.GetPiece(nr, nc);
            if (target == 0 || IsEnemy(board.GetPiece(from / 9, from % 9), target))
            {
                moves.Add(new Move(from, nr * 9 + nc));
            }
        }

        private bool IsEnemy(sbyte p1, sbyte p2) => (p1 > 0 && p2 < 0) || (p1 < 0 && p2 > 0);
    }
}