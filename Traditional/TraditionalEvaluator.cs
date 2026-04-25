using ChineseChessAI.Core;

namespace ChineseChessAI.Traditional
{
    public sealed class TraditionalEvaluator
    {
        private static readonly int[] PieceValues = { 0, 0, 200, 200, 400, 900, 450, 100 };
        public int Evaluate(Board board)
        {
            int redScore = 0;
            int blackScore = 0;
            int totalMaterial = 0;

            for (int index = 0; index < 90; index++)
            {
                sbyte piece = board.GetPiece(index);
                if (piece == 0)
                    continue;

                totalMaterial += PieceValues[Math.Abs(piece)];
                int score = PieceValues[Math.Abs(piece)] + GetPositionBonus(piece, index);
                score += GetShapeBonus(board, piece, index);
                score += GetAttackDefenseBonus(board, piece, index);
                if (piece > 0)
                    redScore += score;
                else
                    blackScore += score;
            }

            redScore += GetKingSafety(board, true);
            blackScore += GetKingSafety(board, false);
            redScore += GetEndgameKingPressure(board, true, totalMaterial);
            blackScore += GetEndgameKingPressure(board, false, totalMaterial);

            int sideRelative = redScore - blackScore;
            return board.IsRedTurn ? sideRelative : -sideRelative;
        }

        private static int GetPositionBonus(sbyte piece, int index)
        {
            int type = Math.Abs(piece);
            bool red = piece > 0;
            int row = index / 9;
            int col = index % 9;
            int forwardRank = red ? 9 - row : row;
            int centerDistance = Math.Abs(col - 4);

            return type switch
            {
                7 => GetPawnBonus(red, row, forwardRank, centerDistance),
                4 => 25 + forwardRank * 4 - centerDistance * 4,
                5 => 20 + MobilityLaneBonus(col),
                6 => 18 + forwardRank * 2 - centerDistance * 2,
                2 => IsHomeAdvisor(red, row, col) ? 12 : 0,
                3 => IsHomeBishop(red, row, col) ? 12 : 0,
                _ => 0
            };
        }

        private static int GetPawnBonus(bool red, int row, int forwardRank, int centerDistance)
        {
            bool crossedRiver = red ? row <= 4 : row >= 5;
            int score = forwardRank * 8 - centerDistance * 4;
            if (crossedRiver)
                score += 70 + forwardRank * 8;
            return score;
        }

        private static int MobilityLaneBonus(int col)
        {
            return col is 0 or 8 ? 0 : 10 - Math.Abs(col - 4) * 2;
        }

        private static bool IsHomeAdvisor(bool red, int row, int col)
        {
            return col is 3 or 5 && (red ? row == 9 : row == 0);
        }

        private static bool IsHomeBishop(bool red, int row, int col)
        {
            return col is 2 or 6 && (red ? row == 9 : row == 0);
        }

        private int GetShapeBonus(Board board, sbyte piece, int index)
        {
            int type = Math.Abs(piece);
            int row = index / 9;
            int col = index % 9;
            bool red = piece > 0;

            return type switch
            {
                5 => CountRookMobility(board, row, col) * 5 + OpenFileBonus(board, red, col),
                6 => CountCannonMobilityAndScreens(board, red, row, col),
                4 => CountKnightMobility(board, row, col) * 7,
                7 => PawnShapeBonus(board, red, row, col),
                _ => 0
            };
        }

        private static int CountRookMobility(Board board, int row, int col)
        {
            int count = 0;
            foreach (var (dr, dc) in Directions())
            {
                for (int step = 1; step < 10; step++)
                {
                    int nr = row + dr * step;
                    int nc = col + dc * step;
                    if (!InBoard(nr, nc))
                        break;
                    count++;
                    if (board.GetPiece(nr, nc) != 0)
                        break;
                }
            }
            return count;
        }

        private static int CountCannonMobilityAndScreens(Board board, bool red, int row, int col)
        {
            int score = 0;
            foreach (var (dr, dc) in Directions())
            {
                bool overScreen = false;
                for (int step = 1; step < 10; step++)
                {
                    int nr = row + dr * step;
                    int nc = col + dc * step;
                    if (!InBoard(nr, nc))
                        break;

                    sbyte target = board.GetPiece(nr, nc);
                    if (!overScreen)
                    {
                        if (target == 0)
                            score += 3;
                        else
                            overScreen = true;
                    }
                    else if (target != 0)
                    {
                        if ((red && target < 0) || (!red && target > 0))
                            score += 30 + PieceValues[Math.Abs(target)] / 25;
                        break;
                    }
                }
            }
            return score;
        }

        private static int CountKnightMobility(Board board, int row, int col)
        {
            int[] dr = { -2, -2, -1, -1, 1, 1, 2, 2 };
            int[] dc = { -1, 1, -2, 2, -2, 2, -1, 1 };
            int[] lr = { -1, -1, 0, 0, 0, 0, 1, 1 };
            int[] lc = { 0, 0, -1, 1, -1, 1, 0, 0 };
            int count = 0;
            for (int i = 0; i < 8; i++)
            {
                int nr = row + dr[i], nc = col + dc[i];
                if (InBoard(nr, nc) && board.GetPiece(row + lr[i], col + lc[i]) == 0)
                    count++;
            }
            return count;
        }

        private static int PawnShapeBonus(Board board, bool red, int row, int col)
        {
            int score = 0;
            sbyte pawn = red ? (sbyte)7 : (sbyte)-7;
            if (col > 0 && board.GetPiece(row, col - 1) == pawn)
                score += 18;
            if (col < 8 && board.GetPiece(row, col + 1) == pawn)
                score += 18;
            int forward = red ? row - 1 : row + 1;
            if (InBoard(forward, col) && board.GetPiece(forward, col) == 0)
                score += 8;
            return score;
        }

        private static int OpenFileBonus(Board board, bool red, int col)
        {
            int ownPawns = 0;
            sbyte pawn = red ? (sbyte)7 : (sbyte)-7;
            for (int row = 0; row < 10; row++)
            {
                if (board.GetPiece(row, col) == pawn)
                    ownPawns++;
            }
            return ownPawns == 0 ? 18 : 0;
        }

        private int GetAttackDefenseBonus(Board board, sbyte piece, int index)
        {
            int attackers = CountAttackers(board, index, piece > 0 ? false : true);
            int defenders = CountAttackers(board, index, piece > 0);
            int value = PieceValues[Math.Abs(piece)];
            int score = defenders * Math.Min(35, value / 25);
            if (attackers > 0)
                score -= attackers * Math.Min(80, value / 12);
            return score;
        }

        private static int CountAttackers(Board board, int targetIndex, bool redAttackers)
        {
            int count = 0;
            for (int from = 0; from < 90; from++)
            {
                sbyte piece = board.GetPiece(from);
                if (piece == 0 || (piece > 0) != redAttackers)
                    continue;
                if (AttacksSquare(board, from, targetIndex, piece))
                    count++;
            }
            return count;
        }

        private static bool AttacksSquare(Board board, int from, int target, sbyte piece)
        {
            if (from == target)
                return false;

            int fr = from / 9, fc = from % 9;
            int tr = target / 9, tc = target % 9;
            int type = Math.Abs(piece);
            bool red = piece > 0;

            switch (type)
            {
                case 1:
                    return fc == tc && IsClearLine(board, fr, fc, tr, tc);
                case 2:
                    return Math.Abs(fr - tr) == 1 && Math.Abs(fc - tc) == 1 && tc >= 3 && tc <= 5 && (red ? tr >= 7 : tr <= 2);
                case 3:
                    return Math.Abs(fr - tr) == 2 && Math.Abs(fc - tc) == 2 &&
                           (red ? tr >= 5 : tr <= 4) &&
                           board.GetPiece((fr + tr) / 2, (fc + tc) / 2) == 0;
                case 4:
                    int dr = tr - fr;
                    int dc = tc - fc;
                    if (!((Math.Abs(dr) == 2 && Math.Abs(dc) == 1) || (Math.Abs(dr) == 1 && Math.Abs(dc) == 2)))
                        return false;
                    int legR = fr + (Math.Abs(dr) == 2 ? Math.Sign(dr) : 0);
                    int legC = fc + (Math.Abs(dc) == 2 ? Math.Sign(dc) : 0);
                    return board.GetPiece(legR, legC) == 0;
                case 5:
                    return (fr == tr || fc == tc) && IsClearLine(board, fr, fc, tr, tc);
                case 6:
                    return (fr == tr || fc == tc) && CountPiecesBetween(board, fr, fc, tr, tc) == 1;
                case 7:
                    if (red)
                    {
                        if (tr == fr - 1 && tc == fc)
                            return true;
                        return fr <= 4 && tr == fr && Math.Abs(tc - fc) == 1;
                    }
                    if (tr == fr + 1 && tc == fc)
                        return true;
                    return fr >= 5 && tr == fr && Math.Abs(tc - fc) == 1;
                default:
                    return false;
            }
        }

        private static bool IsClearLine(Board board, int fr, int fc, int tr, int tc)
        {
            return CountPiecesBetween(board, fr, fc, tr, tc) == 0;
        }

        private static int CountPiecesBetween(Board board, int fr, int fc, int tr, int tc)
        {
            if (fr != tr && fc != tc)
                return int.MaxValue;
            int dr = Math.Sign(tr - fr);
            int dc = Math.Sign(tc - fc);
            int count = 0;
            int r = fr + dr;
            int c = fc + dc;
            while (r != tr || c != tc)
            {
                if (board.GetPiece(r, c) != 0)
                    count++;
                r += dr;
                c += dc;
            }
            return count;
        }

        private int GetKingSafety(Board board, bool red)
        {
            int king = FindKing(board, red);
            if (king < 0)
                return -10_000;

            int row = king / 9;
            int col = king % 9;
            int score = 0;
            if (board.GetPiece(red ? 9 : 0, 3) == (red ? 2 : -2))
                score += 20;
            if (board.GetPiece(red ? 9 : 0, 5) == (red ? 2 : -2))
                score += 20;
            if (board.GetPiece(red ? 9 : 0, 2) == (red ? 3 : -3))
                score += 12;
            if (board.GetPiece(red ? 9 : 0, 6) == (red ? 3 : -3))
                score += 12;
            if (CountAttackers(board, king, !red) > 0)
                score -= 120;
            score -= Math.Abs(col - 4) * 10;
            score -= red ? Math.Max(0, 9 - row) * 6 : row * 6;
            return score;
        }

        private static int GetEndgameKingPressure(Board board, bool red, int totalMaterial)
        {
            if (totalMaterial > 3200)
                return 0;

            int ownKing = FindKing(board, red);
            int enemyKing = FindKing(board, !red);
            if (ownKing < 0 || enemyKing < 0)
                return 0;

            int ownRow = ownKing / 9, ownCol = ownKing % 9;
            int enemyRow = enemyKing / 9, enemyCol = enemyKing % 9;
            int distance = Math.Abs(ownRow - enemyRow) + Math.Abs(ownCol - enemyCol);
            return Math.Max(0, 60 - distance * 8);
        }

        private static int FindKing(Board board, bool red)
        {
            sbyte king = red ? (sbyte)1 : (sbyte)-1;
            for (int i = 0; i < 90; i++)
            {
                if (board.GetPiece(i) == king)
                    return i;
            }
            return -1;
        }

        private static IEnumerable<(int dr, int dc)> Directions()
        {
            yield return (-1, 0);
            yield return (1, 0);
            yield return (0, -1);
            yield return (0, 1);
        }

        private static bool InBoard(int row, int col)
        {
            return row >= 0 && row < 10 && col >= 0 && col < 9;
        }
    }
}
