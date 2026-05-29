namespace ChineseChessAI.Core
{
    public class MoveGenerator
    {
        private const int ROWS = 10;
        private const int COLS = 9;
        private readonly Action<string>? _statusChanged;
        private int _lastStatusTick;

        // 长杀搜索 transposition table，跨 IsThreateningToMate 调用共享。
        // MoveGenerator 在 MCTSEngine 内被多线程并发使用，故用 ConcurrentDictionary。
        private readonly System.Collections.Concurrent.ConcurrentDictionary<(ulong Hash, bool IsRedAttacker, int Depth), bool> _mateMemo = new();
        private const int MateMemoMaxSize = 200_000;

        // 热路径线程本地缓冲：禁手判定不会自身嵌套（GenerateLegalMoves 内层都传 skipPerpetualCheck=true），
        // 故同线程内 IsForbiddenPerpetualMove → SnapshotAttackPairs → IsChasing 串行使用同一组 buffer 安全。
        [ThreadStatic] private static List<Move>? _tlsMoves;
        [ThreadStatic] private static HashSet<int>? _tlsPreAttacks;
        [ThreadStatic] private static HashSet<int>? _tlsAffected;

        private static List<Move> RentTlsMoves()
        {
            var l = _tlsMoves ??= new List<Move>(64);
            l.Clear();
            return l;
        }

        private static HashSet<int> RentTlsPreAttacks()
        {
            var s = _tlsPreAttacks ??= new HashSet<int>(64);
            s.Clear();
            return s;
        }

        private static HashSet<int> RentTlsAffected()
        {
            var s = _tlsAffected ??= new HashSet<int>(8);
            s.Clear();
            return s;
        }

        public MoveGenerator(Action<string>? statusChanged = null)
        {
            _statusChanged = statusChanged;
        }

        public string GetMoveValidationResult(Board board, Move move, bool skipPerpetualCheck = false, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pseudoMoves = new List<Move>(64);
            sbyte piece = board.GetPiece(move.From);
            if (piece == 0)
                return "起点无棋子";
            bool isRed = piece > 0;
            if (isRed != board.IsRedTurn)
                return "走子方错误";

            // 1. 检查基础物理走法
            GeneratePieceMoves(board, move.From, piece, pseudoMoves);
            if (!pseudoMoves.Any(m => m.From == move.From && m.To == move.To))
                return "违规移动";

            // 2. 检查是否导致自方被将军（送将）
            sbyte captured = board.PerformMoveInternal(move.From, move.To);
            bool safe = IsKingSafe(board, isRed);
            board.UndoMoveInternal(move.From, move.To, captured);
            if (!safe)
                return "自方王受威胁 (送将)";

            // 3. 检查禁手规则（长打/长捉）
            if (!skipPerpetualCheck && board.WillCauseThreefoldRepetition(move.From, move.To))
            {
                if (IsForbiddenPerpetualMove(board, move, cancellationToken))
                    return "禁手 (违规长打/长捉)";
            }

            return "合法";
        }

        /// <summary>
        /// 生成当前局面的所有合法走法（已过滤送将、飞将及违反长将/长捉规则的走法）
        /// </summary>
        public List<Move> GenerateLegalMoves(Board board, bool skipPerpetualCheck = false, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pseudoMoves = new List<Move>(64);
            bool isRed = board.IsRedTurn;

            for (int i = 0; i < 90; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
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
                cancellationToken.ThrowIfCancellationRequested();
                sbyte captured = board.PerformMoveInternal(move.From, move.To);
                bool safe = IsKingSafe(board, isRed);
                board.UndoMoveInternal(move.From, move.To, captured);

                if (!safe)
                    continue;

                if (!skipPerpetualCheck && board.WillCauseThreefoldRepetition(move.From, move.To))
                {
                    if (IsForbiddenPerpetualMove(board, move, cancellationToken))
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

        private bool IsForbiddenPerpetualMove(Board board, Move move, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // 走子前先快照攻击方所有"对敌方非豁免子"的攻击 pair (attackerPos, targetPos)，
            // 用于 IsChasing 区分"既得攻击"与"新生攻击"——只有新生攻击才计入捉。
            bool isRedAttacker = board.IsRedTurn;
            var preMoveAttacks = RentTlsPreAttacks();
            FillAttackPairs(board, isRedAttacker, preMoveAttacks);

            board.Push(move.From, move.To);
            int count = board.GetRepetitionCount();
            bool isForbidden = false;
            if (count >= 3)
            {
                // 长将、长捉、长杀均视为禁手
                ReportStatus("正在检查长将/长打禁着与强制杀威胁...");
                bool isChecking = IsChecking(board, !board.IsRedTurn);
                if (isChecking)
                {
                    isForbidden = true;
                }
                else
                {
                    bool isChasing = IsChasing(board, move, preMoveAttacks);
                    if (isChasing)
                    {
                        isForbidden = true;
                    }
                    else
                    {
                        ReportStatus("正在检查长将/长打禁着与强制杀威胁...");
                        isForbidden = IsThreateningToMate(board, !board.IsRedTurn, cancellationToken);
                    }
                }
            }
            board.Pop();
            return isForbidden;
        }

        /// <summary>
        /// 写入当前局面下，攻击方对所有敌方"可被捉子"的攻击对 (attackerPos*100+targetPos)到提供的 set。
        /// 仅记录满足 IsRealChase 入口豁免条件外的目标（将士象 / 未过河兵卒被排除前置过滤）。
        /// </summary>
        private void FillAttackPairs(Board board, bool isRedAttacker, HashSet<int> set)
        {
            var moves = RentTlsMoves();
            for (int i = 0; i < 90; i++)
            {
                sbyte p = board.GetPiece(i);
                if (p == 0)
                    continue;
                if ((isRedAttacker && p < 0) || (!isRedAttacker && p > 0))
                    continue;

                moves.Clear();
                GeneratePieceMoves(board, i, p, moves);
                foreach (var m in moves)
                {
                    sbyte target = board.GetPiece(m.To);
                    if (target == 0)
                        continue;
                    if ((isRedAttacker && target > 0) || (!isRedAttacker && target < 0))
                        continue;

                    int targetType = Math.Abs(target);
                    if (targetType <= 3)
                        continue; // 将士象不计入捉
                    if (targetType == 7)
                    {
                        bool isRedTarget = target > 0;
                        int row = m.To / 9;
                        if (isRedTarget ? (row >= 5) : (row <= 4))
                            continue; // 未过河兵卒不计入捉
                    }
                    set.Add(i * 100 + m.To);
                }
            }
        }

        private void ReportStatus(string message)
        {
            if (_statusChanged == null)
                return;

            int now = Environment.TickCount;
            if (unchecked(now - _lastStatusTick) < 1000)
                return;

            _lastStatusTick = now;
            _statusChanged(message);
        }

        public bool IsChecking(Board board, bool isRedAttacker)
        {
            return !IsKingSafe(board, !isRedAttacker);
        }

        /// <summary>
        /// 判断走子是否构成"捉"的威胁。
        /// 仅当攻击 (attackerPos, targetPos) 在走子前不存在（新生攻击或抽吃）时才计入捉，
        /// 排除既得既存攻击导致的误判。同时过滤"假打"（被牵制子的虚假攻击）。
        /// </summary>
        private bool IsChasing(Board board, Move move, HashSet<int> preMoveAttacks)
        {
            // 在执行 move 之后的局面判断
            bool isRedAttacker = !board.IsRedTurn;
            int attackerPos = move.To;
            sbyte attacker = board.GetPiece(attackerPos);
            if (attacker == 0 || (isRedAttacker ? attacker < 0 : attacker > 0))
                return false;

            // 1. 走动棋子自身的捉：pre 中该棋子在 move.From，故查 (move.From, m.To) 是否已存在
            var attacks = RentTlsMoves();
            GeneratePieceMoves(board, attackerPos, attacker, attacks);
            foreach (var m in attacks)
            {
                sbyte target = board.GetPiece(m.To);
                if (target == 0 || (isRedAttacker ? target > 0 : target < 0))
                    continue;

                int preKey = move.From * 100 + m.To;
                if (preMoveAttacks.Contains(preKey))
                    continue; // 既得攻击，本步未新增捉势

                if (!IsAttackLegal(board, attackerPos, m.To, isRedAttacker))
                    continue; // 假打：攻击者被牵制，吃子即送将

                if (IsRealChase(board, attackerPos, m.To))
                    return true;
            }

            // 2. 抽吃 / 闪击：本步可能改变攻击集的友方子（沿 from/to 横纵线上的车王炮 + 邻接 4 格友方马）
            //    其它子（士象兵、远处子）的攻击集本步不变，已被 preMoveAttacks 全场快照覆盖。
            var affected = RentTlsAffected();
            CollectAffectedFriendlies(board, move.From, move.To, isRedAttacker, attackerPos, affected);

            // 注：foreach 内调用 GeneratePieceMoves 复用 _tlsMoves（与上方 attacks 同一缓冲），
            //    第一段 attacks foreach 已结束、attacks 不再读，复用安全。
            foreach (int i in affected)
            {
                sbyte otherAttacker = board.GetPiece(i);
                // affected 集合在加入时已校验同色，但保留 sanity 检查
                if (otherAttacker == 0 || (isRedAttacker ? otherAttacker < 0 : otherAttacker > 0))
                    continue;

                var otherAttacks = RentTlsMoves();
                GeneratePieceMoves(board, i, otherAttacker, otherAttacks);
                foreach (var m in otherAttacks)
                {
                    sbyte target = board.GetPiece(m.To);
                    if (target == 0 || (isRedAttacker ? target > 0 : target < 0))
                        continue;

                    int preKey = i * 100 + m.To;
                    if (preMoveAttacks.Contains(preKey))
                        continue; // 既得攻击

                    if (!IsAttackLegal(board, i, m.To, isRedAttacker))
                        continue; // 假打过滤

                    if (IsRealChase(board, i, m.To))
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// 收集本步走子可能改变其攻击集的"友方子"集合。
        /// 几何分析：沿 from / to 的横纵 4 方向第一/二个非空若是友方车/王/炮 → 视野受影响；
        /// from / to 的邻接 4 格上若有友方马 → 该马的某些走法的"马腿"是 from/to，攻击集变化。
        /// 排除 attacker 自己（excludePos = move.To，已在第一段处理）。
        /// 非线性子（士、象、兵）的攻击集不受单步移动影响，故不纳入。
        /// </summary>
        private void CollectAffectedFriendlies(Board board, int from, int to, bool isRedAttacker, int excludePos, HashSet<int> affected)
        {
            int[] linDr = { -1, 1, 0, 0 };
            int[] linDc = { 0, 0, -1, 1 };
            int[] adjDr = { -1, 1, 0, 0 };
            int[] adjDc = { 0, 0, -1, 1 };

            for (int o = 0; o < 2; o++)
            {
                int origin = (o == 0) ? from : to;
                int r = origin / 9, c = origin % 9;

                // 沿 origin 的横/纵 4 方向：第一非空若友方车/王/炮加入；第二非空若友方炮加入
                for (int dir = 0; dir < 4; dir++)
                {
                    int count = 0;
                    for (int step = 1; step < 10; step++)
                    {
                        int nr = r + linDr[dir] * step;
                        int nc = c + linDc[dir] * step;
                        if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                            break;
                        int idx = nr * 9 + nc;
                        sbyte p = board.GetPiece(idx);
                        if (p == 0)
                            continue;
                        count++;
                        if (idx != excludePos)
                        {
                            bool sameColor = isRedAttacker ? p > 0 : p < 0;
                            if (sameColor)
                            {
                                int t = Math.Abs(p);
                                if (t == 5 || t == 1 || t == 6)
                                    affected.Add(idx);
                            }
                        }
                        if (count >= 2)
                            break;
                    }
                }

                // 邻接 4 格上的友方马（以 origin 为马腿之一的马）
                for (int i = 0; i < 4; i++)
                {
                    int hr = r + adjDr[i];
                    int hc = c + adjDc[i];
                    if (hr < 0 || hr >= ROWS || hc < 0 || hc >= COLS)
                        continue;
                    int hidx = hr * 9 + hc;
                    if (hidx == excludePos)
                        continue;
                    sbyte p = board.GetPiece(hidx);
                    if (Math.Abs(p) != 4)
                        continue;
                    bool sameColor = isRedAttacker ? p > 0 : p < 0;
                    if (sameColor)
                        affected.Add(hidx);
                }
            }
        }

        /// <summary>
        /// 验证 from→to 这步攻击是否合法（攻击方完成捕获后，自方将不被攻击）。
        /// 用于过滤"假打"——被牵制子的虚假威胁不计入捉。
        /// </summary>
        private bool IsAttackLegal(Board board, int from, int to, bool isRedAttacker)
        {
            sbyte captured = board.PerformMoveInternal(from, to);
            try
            {
                return IsKingSafe(board, isRedAttacker);
            }
            finally
            {
                board.UndoMoveInternal(from, to, captured);
            }
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
            if (targetType <= 3)
                return false;

            // 未过河兵卒不计入常捉
            if (targetType == 7)
            {
                bool isRedTarget = target > 0;
                int row = to / 9;
                if (isRedTarget ? (row >= 5) : (row <= 4))
                    return false;
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

            if (!protectedTarget)
                return true;
            if (attackerVal < targetVal)
                return true;

            return false;
        }

        /// <summary>
        /// 判断 pos 上的棋子是否被同色友军保护（对方若吃掉它，本方有子可以回吃）。
        /// 反向几何查找：从 pos 出发反推哪些位置上的友军能攻击/保护 pos，避免 90 格全扫。
        /// </summary>
        private bool IsProtected(Board board, int pos)
        {
            sbyte target = board.GetPiece(pos);
            if (target == 0)
                return false;
            bool isRed = target > 0;
            int r = pos / 9, c = pos % 9;

            // A. 横/纵线扫描：第一个非空若是己方车 → 保护；step==1 且是己方王 → 保护；
            //    第二个非空（炮架后）若是己方炮 → 保护
            int[] linDr = { -1, 1, 0, 0 };
            int[] linDc = { 0, 0, -1, 1 };
            for (int dir = 0; dir < 4; dir++)
            {
                int count = 0;
                for (int step = 1; step < 10; step++)
                {
                    int nr = r + linDr[dir] * step, nc = c + linDc[dir] * step;
                    if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS)
                        break;
                    sbyte p = board.GetPiece(nr, nc);
                    if (p == 0)
                        continue;

                    count++;
                    if (count == 1)
                    {
                        bool sameColor = isRed ? p > 0 : p < 0;
                        if (sameColor)
                        {
                            int t = Math.Abs(p);
                            if (t == 5)
                                return true; // 车
                            if (t == 1 && step == 1)
                                return true; // 王（仅相邻一格）
                        }
                        // 不论敌友，下一格起作为潜在炮架
                    }
                    else if (count == 2)
                    {
                        bool sameColor = isRed ? p > 0 : p < 0;
                        if (sameColor && Math.Abs(p) == 6)
                            return true; // 隔一架的己方炮
                        break; // 第三个棋子起再无线性威胁
                    }
                }
            }

            // B. 马的保护：8 个"日字"起点上是否有己方马，且其马腿空
            int[] knDr = { -2, -2, -1, -1, 1, 1, 2, 2 };
            int[] knDc = { -1, 1, -2, 2, -2, 2, -1, 1 };
            for (int i = 0; i < 8; i++)
            {
                int kr = r + knDr[i], kc = c + knDc[i];
                if (kr < 0 || kr >= ROWS || kc < 0 || kc >= COLS)
                    continue;
                sbyte p = board.GetPiece(kr, kc);
                if (Math.Abs(p) != 4)
                    continue;
                if (isRed ? p < 0 : p > 0)
                    continue; // 敌方马
                // 马腿：紧挨着马朝 pos 方向的格子
                int ddr = r - kr, ddc = c - kc;
                int legR, legC;
                if (Math.Abs(ddr) == 2)
                {
                    legR = kr + ddr / 2;
                    legC = kc;
                }
                else
                {
                    legR = kr;
                    legC = kc + ddc / 2;
                }
                if (board.GetPiece(legR, legC) == 0)
                    return true;
            }

            // C. 兵的保护
            if (isRed)
            {
                // 红兵向上走（forwardR = r-1），故红兵在 (r+1, c) 时能保护 (r, c)
                if (r + 1 < ROWS && board.GetPiece(r + 1, c) == 7)
                    return true;
                // 红兵过河（自己 row<=4）后可横走，故 (r, c±1) 处的红兵能保护 (r, c) 当 r<=4
                if (r <= 4)
                {
                    if (c - 1 >= 0 && board.GetPiece(r, c - 1) == 7)
                        return true;
                    if (c + 1 < COLS && board.GetPiece(r, c + 1) == 7)
                        return true;
                }
            }
            else
            {
                if (r - 1 >= 0 && board.GetPiece(r - 1, c) == -7)
                    return true;
                if (r >= 5)
                {
                    if (c - 1 >= 0 && board.GetPiece(r, c - 1) == -7)
                        return true;
                    if (c + 1 < COLS && board.GetPiece(r, c + 1) == -7)
                        return true;
                }
            }

            // D. 士的保护：斜一格且自己在九宫
            int[] adDr = { -1, -1, 1, 1 };
            int[] adDc = { -1, 1, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int ar = r + adDr[i], ac = c + adDc[i];
                if (ac < 3 || ac > 5)
                    continue;
                if (isRed && (ar < 7 || ar > 9))
                    continue;
                if (!isRed && (ar < 0 || ar > 2))
                    continue;
                sbyte p = board.GetPiece(ar, ac);
                if (Math.Abs(p) == 2 && (isRed ? p > 0 : p < 0))
                    return true;
            }

            // E. 象的保护：田字且象眼空、象在己方半盘
            int[] biDr = { -2, -2, 2, 2 };
            int[] biDc = { -2, 2, -2, 2 };
            int[] biPr = { -1, -1, 1, 1 };
            int[] biPc = { -1, 1, -1, 1 };
            for (int i = 0; i < 4; i++)
            {
                int br = r + biDr[i], bc = c + biDc[i];
                if (br < 0 || br >= ROWS || bc < 0 || bc >= COLS)
                    continue;
                if (isRed && br < 5)
                    continue;
                if (!isRed && br > 4)
                    continue;
                if (board.GetPiece(r + biPr[i], c + biPc[i]) != 0)
                    continue; // 象眼被堵
                sbyte p = board.GetPiece(br, bc);
                if (Math.Abs(p) == 3 && (isRed ? p > 0 : p < 0))
                    return true;
            }

            return false;
        }

        // 长杀禁手入口：检查攻击方是否已形成强制杀势（最多向前搜索 MateSearchDepth 步攻击）
        private const int MateSearchDepth = 5;
        private const int MateSearchNodeLimit = 20_000;
        private static readonly TimeSpan MateSearchTimeLimit = TimeSpan.FromMilliseconds(250);

        public bool IsThreateningToMate(Board board, bool isRedAttacker, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // 调用时机：IsForbiddenPerpetualMove 已 Push(攻击方着法)，board.IsRedTurn = 防守方。
            // 正确语义：对"防守方所有合法应手"（AND 节点）逐一检查，每个应手之后
            // 攻击方是否仍有强制杀势（HasForcedKill = OR 节点）。
            // memo 用类共享 _mateMemo，跨调用复用搜索结果（同形面命中率极高）。
            if (_mateMemo.Count > MateMemoMaxSize)
                _mateMemo.Clear();

            var budget = new MateSearchBudget(MateSearchNodeLimit, MateSearchTimeLimit);
            var defenderMoves = GenerateLegalMoves(board, skipPerpetualCheck: true, cancellationToken);
            if (defenderMoves.Count == 0)
                return true;

            foreach (var defenderMove in defenderMoves)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!budget.TryConsume())
                    return false;

                board.Push(defenderMove.From, defenderMove.To);
                try
                {
                    if (HasForcedKill(board, isRedAttacker, MateSearchDepth, budget, cancellationToken) != true)
                        return false;
                }
                finally
                {
                    board.Pop();
                }
            }

            return true;
        }

        private bool? HasForcedKill(
            Board board,
            bool isRedAttacker,
            int depth,
            MateSearchBudget budget,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!budget.TryConsume())
                return null;

            if (depth <= 0)
                return false;

            var key = (board.CurrentHash, isRedAttacker, depth);
            if (_mateMemo.TryGetValue(key, out bool cached))
                return cached;

            bool sawBudgetExhaustedLine = false;
            foreach (var attackerMove in GenerateLegalMoves(board, skipPerpetualCheck: true, cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!budget.TryConsume())
                    return null;

                board.Push(attackerMove.From, attackerMove.To);
                try
                {
                    var defenderMoves = GenerateLegalMoves(board, skipPerpetualCheck: true, cancellationToken);
                    if (defenderMoves.Count == 0)
                    {
                        _mateMemo[key] = true;
                        return true;
                    }

                    bool defenderHasEscape = false;
                    bool defenderLineUnknown = false;
                    foreach (var defenderMove in defenderMoves)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (!budget.TryConsume())
                            return null;

                        board.Push(defenderMove.From, defenderMove.To);
                        try
                        {
                            bool? childResult = HasForcedKill(board, isRedAttacker, depth - 1, budget, cancellationToken);
                            if (childResult == null)
                            {
                                defenderLineUnknown = true;
                                break;
                            }

                            if (childResult != true)
                            {
                                defenderHasEscape = true;
                                break;
                            }
                        }
                        finally
                        {
                            board.Pop();
                        }
                    }

                    if (defenderLineUnknown)
                    {
                        sawBudgetExhaustedLine = true;
                        continue;
                    }

                    if (!defenderHasEscape)
                    {
                        _mateMemo[key] = true;
                        return true;
                    }
                }
                finally
                {
                    board.Pop();
                }
            }

            if (sawBudgetExhaustedLine)
                return null;

            _mateMemo[key] = false;
            return false;
        }

        // 棋例分值（亚洲象棋联合会规则，用于"捉/兑/闲"判定，与子力评估分离）
        // 车=4，马=炮=2（等价子互捉判兑），过河兵卒=1，将帅特殊高分
        private sealed class MateSearchBudget
        {
            private readonly int _nodeLimit;
            private readonly TimeSpan _timeLimit;
            private readonly System.Diagnostics.Stopwatch _stopwatch = System.Diagnostics.Stopwatch.StartNew();
            private int _nodes;

            public MateSearchBudget(int nodeLimit, TimeSpan timeLimit)
            {
                _nodeLimit = nodeLimit;
                _timeLimit = timeLimit;
            }

            public bool TryConsume()
            {
                if (Interlocked.Increment(ref _nodes) > _nodeLimit)
                    return false;

                return _stopwatch.Elapsed <= _timeLimit;
            }
        }

        private int GetPieceValue(sbyte piece)
        {
            switch (Math.Abs(piece))
            {
                case 1:
                    return 1000; // 帅
                case 5:
                    return 4;    // 俥
                case 4:
                    return 2;    // 傌
                case 6:
                    return 2;    // 炮（与马等价）
                case 7:
                    return 1;    // 卒 (过河)
                default:
                    return 1;
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
