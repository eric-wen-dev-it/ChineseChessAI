namespace ChineseChessAI.Core
{
    /// <summary>
    /// 重复局面裁决结果。禁手方(长将/长捉)判负。
    /// </summary>
    public enum PerpetualVerdict
    {
        /// <summary>非单方面禁手(一将一还将/一捉一还捉/无禁手)→ 按普通规则判和。</summary>
        None,
        /// <summary>红方长将或长捉,红负。</summary>
        RedLoses,
        /// <summary>黑方长将或长捉,黑负。</summary>
        BlackLoses
    }

    /// <summary>
    /// 逐手记录"走子方 / 是否将军 / 是否捉子 / 走后局面哈希",在三次重复触发时按最近一个
    /// 重复循环判定长将/长捉归属。规则:循环内某一方每一手都在将军(且对方不是)→长将判该方负;
    /// 都不长将时,某一方每一手都在捉(且对方不是)→长捉判该方负;否则判和(None)。
    /// 走一步 irreversible(吃子/进兵)后重复窗口失效,须 Reset。
    /// </summary>
    public sealed class PerpetualRepetitionTracker
    {
        private readonly List<bool> _moverRed = new();
        private readonly List<bool> _gaveCheck = new();
        private readonly List<bool> _wasChase = new();
        private readonly List<ulong> _hash = new();

        public void Reset()
        {
            _moverRed.Clear();
            _gaveCheck.Clear();
            _wasChase.Clear();
            _hash.Clear();
        }

        public void Record(bool moverRed, bool gaveCheck, bool wasChase, ulong hashAfterMove)
        {
            _moverRed.Add(moverRed);
            _gaveCheck.Add(gaveCheck);
            _wasChase.Add(wasChase);
            _hash.Add(hashAfterMove);
        }

        /// <summary>
        /// 在刚 Record 完、且该局面已达三次重复时调用:回看最近一个重复循环判禁手归属。
        /// </summary>
        public PerpetualVerdict Classify()
        {
            int last = _hash.Count - 1;
            if (last < 0)
                return PerpetualVerdict.None;

            ulong h = _hash[last];
            int prev = -1;
            for (int i = last - 1; i >= 0; i--)
            {
                if (_hash[i] == h)
                {
                    prev = i;
                    break;
                }
            }
            if (prev < 0)
                return PerpetualVerdict.None;

            // 窗口 [prev+1 .. last] 恰为最近一个完整重复循环。
            int start = prev + 1;
            bool redAny = false, redAllCheck = true, redAllChase = true;
            bool blackAny = false, blackAllCheck = true, blackAllChase = true;
            for (int i = start; i <= last; i++)
            {
                if (_moverRed[i])
                {
                    redAny = true;
                    if (!_gaveCheck[i]) redAllCheck = false;
                    if (!_wasChase[i]) redAllChase = false;
                }
                else
                {
                    blackAny = true;
                    if (!_gaveCheck[i]) blackAllCheck = false;
                    if (!_wasChase[i]) blackAllChase = false;
                }
            }

            // 长将优先:某方循环内每手皆将军,对方不是。
            bool redPerpCheck = redAny && redAllCheck;
            bool blackPerpCheck = blackAny && blackAllCheck;
            if (redPerpCheck && !blackPerpCheck) return PerpetualVerdict.RedLoses;
            if (blackPerpCheck && !redPerpCheck) return PerpetualVerdict.BlackLoses;
            if (redPerpCheck && blackPerpCheck) return PerpetualVerdict.None; // 一将一还将 = 和

            // 长捉:某方循环内每手皆捉,对方不是。
            bool redPerpChase = redAny && redAllChase;
            bool blackPerpChase = blackAny && blackAllChase;
            if (redPerpChase && !blackPerpChase) return PerpetualVerdict.RedLoses;
            if (blackPerpChase && !redPerpChase) return PerpetualVerdict.BlackLoses;

            return PerpetualVerdict.None;
        }
    }
}
