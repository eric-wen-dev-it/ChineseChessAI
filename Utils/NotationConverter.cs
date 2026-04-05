using ChineseChessAI.Core;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ChineseChessAI.Utils
{
    public static class NotationConverter
    {
        /// <summary>
        /// 万能转换器：将任意记谱法（WXF代数/UCCI）统一转换为标准 UCCI 格式 (如 "h2e2")
        /// </summary>
        public static string? ConvertToUcci(Board board, string rawMove, MoveGenerator generator)
        {
            rawMove = rawMove.Trim().Replace("=", "."); // 兼容某些用 = 代替 . 的平移记法

            // 1. 如果已经是 UCCI 格式 (如 h2e2, a9b9)，直接返回
            if (rawMove.Length == 4 &&
                rawMove[0] >= 'a' && rawMove[0] <= 'i' &&
                rawMove[1] >= '0' && rawMove[1] <= '9' &&
                rawMove[2] >= 'a' && rawMove[2] <= 'i' &&
                rawMove[3] >= '0' && rawMove[3] <= '9')
            {
                return rawMove.ToLower();
            }

            // 2. 否则按 WXF 代数谱进行物理推导
            Move? move = ParseWxfMove(board, rawMove, generator);
            if (move == null)
                return null;

            // 将推导出的物理走法转成 UCCI
            return MoveToUcci(move.Value);
        }

        /// <summary>
        /// UCCI 字符串 (如 "h2e2") 转内部物理 Move 对象
        /// </summary>
        public static Move? UcciToMove(string ucci)
        {
            if (string.IsNullOrEmpty(ucci) || ucci.Length != 4)
                return null;
            ucci = ucci.ToLower();

            int fC = ucci[0] - 'a';
            int fR = 9 - (ucci[1] - '0');
            int tC = ucci[2] - 'a';
            int tR = 9 - (ucci[3] - '0');

            // 越界保护
            if (fC < 0 || fC > 8 || fR < 0 || fR > 9 || tC < 0 || tC > 8 || tR < 0 || tR > 9)
                return null;

            return new Move(fR * 9 + fC, tR * 9 + tC);
        }

        /// <summary>
        /// 内部物理 Move 对象转 UCCI 字符串
        /// </summary>
        public static string MoveToUcci(Move move)
        {
            char fC = (char)('a' + (move.From % 9));
            char fR = (char)('0' + (9 - (move.From / 9)));
            char tC = (char)('a' + (move.To % 9));
            char tR = (char)('0' + (9 - (move.To / 9)));
            return $"{fC}{fR}{tC}{tR}";
        }

        /// <summary>
        /// WXF 智能代数谱推导引擎
        /// </summary>
        private static Move? ParseWxfMove(Board board, string wxf, MoveGenerator generator)
        {
            var legalMoves = generator.GenerateLegalMoves(board);
            bool isRed = board.IsRedTurn;
            var candidates = new List<Move>();

            wxf = wxf.ToUpper();

            foreach (var move in legalMoves)
            {
                sbyte piece = board.GetPiece(move.From);
                int type = Math.Abs(piece);
                char pieceChar = type switch
                {
                    1 => 'K',
                    2 => 'A',
                    3 => 'E',
                    4 => 'H',
                    5 => 'R',
                    6 => 'C',
                    7 => 'P',
                    _ => '?'
                };

                // 兼容某些用 B (Bishop) 代替 E (Elephant) 的谱
                if (type == 3 && !wxf.Contains('E') && wxf.Contains('B'))
                    pieceChar = 'B';

                if (!wxf.Contains(pieceChar))
                    continue;

                int fromRow = move.From / 9, fromCol = move.From % 9;
                int toRow = move.To / 9, toCol = move.To % 9;

                int startFile = isRed ? (9 - fromCol) : (fromCol + 1);
                int endFile = isRed ? (9 - toCol) : (toCol + 1);

                char direction = fromRow == toRow ? '.' : ((isRed && toRow < fromRow) || (!isRed && toRow > fromRow) ? '+' : '-');
                if (!wxf.Contains(direction))
                    continue;

                int endValue = (type == 2 || type == 3 || type == 4 || direction == '.') ? endFile : Math.Abs(toRow - fromRow);

                var digits = wxf.Where(char.IsDigit).ToArray();
                if (digits.Length == 0)
                    continue;

                int expectedEndVal = digits.Last() - '0';
                if (endValue != expectedEndVal)
                    continue;

                if (digits.Length >= 2)
                {
                    int expectedStartFile = digits.First() - '0';
                    if (startFile != expectedStartFile)
                        continue;
                }

                candidates.Add(move);
            }

            if (candidates.Count == 1)
                return candidates[0];

            if (candidates.Count > 1)
            {
                bool isFront = wxf.StartsWith("+") || wxf.StartsWith("F");
                bool isBack = wxf.StartsWith("-") || wxf.StartsWith("B");

                if (isFront || isBack)
                {
                    var sorted = candidates.OrderBy(m => m.From / 9).ToList();
                    if (isRed)
                        return isFront ? sorted.First() : sorted.Last();
                    else
                        return isFront ? sorted.Last() : sorted.First();
                }

                // 【核心修复】：没有说明前后，且有多个候选，属于严重歧义的残缺谱！
                // 绝不能默认返回 candidates[0]，直接返回 null 让外层丢弃这局脏数据。
                return null;
            }
            return null;
        }
    }
}