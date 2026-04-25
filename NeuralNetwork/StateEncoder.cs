using ChineseChessAI.Core;
using System.Buffers; // 必须引用：用于 ArrayPool
using TorchSharp;
using static TorchSharp.torch;

namespace ChineseChessAI.NeuralNetwork
{
    public static class StateEncoder
    {
        // 预计算数据大小：14通道 * 10行 * 9列 = 1260
        private const int DataSize = 14 * 10 * 9;

        // 【终极微操优化】：将每次都要 new 的数组提取为静态只读，彻底消灭 GC 垃圾！
        private static readonly long[] FlipDims = new long[] { 1, 2 };

        /// <summary>
        /// 【优化 P3 #9】：直接对稀疏策略进行翻转，效率比全量遍历 8100 提高数百倍。
        /// </summary>
        public static ActionProb[] FlipPolicySparse(ActionProb[] sparsePi)
        {
            if (sparsePi == null)
                return Array.Empty<ActionProb>();
            var flipped = new ActionProb[sparsePi.Length];
            for (int i = 0; i < sparsePi.Length; i++)
            {
                int originalIdx = sparsePi[i].Index;
                int from = originalIdx / 90, to = originalIdx % 90;
                int r1 = from / 9, c1 = from % 9, r2 = to / 9, c2 = to % 9;

                // 180 度中心对称翻转：(r, c) -> (9-r, 8-c)
                int nf = ((9 - r1) * 9 + (8 - c1)) * 90 + ((9 - r2) * 9 + (8 - c2));
                flipped[i] = new ActionProb(nf, sparsePi[i].Prob);
            }
            return flipped;
        }

        /// <summary>
        /// 将策略数组做 180 度翻转，用于黑方视角与红方视角之间的互相转换。
        /// </summary>
        public static float[] FlipPolicy(float[] originalPi)
        {
            float[] flippedPi = new float[8100];
            // 依然保留稠密版以兼容旧接口，但逻辑相同
            for (int i = 0; i < 8100; i++)
            {
                if (originalPi[i] <= 0)
                    continue;
                int from = i / 90, to = i % 90;
                int r1 = from / 9, c1 = from % 9, r2 = to / 9, c2 = to % 9;
                int idx_f = ((9 - r1) * 9 + (8 - c1)) * 90 + ((9 - r2) * 9 + (8 - c2));
                if (idx_f >= 0 && idx_f < 8100)
                    flippedPi[idx_f] = originalPi[i];
            }
            return flippedPi;
        }

        public static Tensor Encode(Board board)
        {
            // 1. 从内存池租借数组
            float[] data = ArrayPool<float>.Shared.Rent(DataSize);

            try
            {
                // 2. 清空数组
                Array.Clear(data, 0, DataSize);

                bool isRedTurn = board.IsRedTurn;

                // --- 填值逻辑 ---
                for (int r = 0; r < 10; r++)
                {
                    for (int c = 0; c < 9; c++)
                    {
                        sbyte piece = board.GetPiece(r, c);
                        if (piece == 0)
                            continue;

                        bool isRedPiece = piece > 0;
                        int pieceType = Math.Abs(piece);
                        int planeIndex;

                        if (isRedTurn == isRedPiece)
                            planeIndex = pieceType - 1;
                        else
                            planeIndex = pieceType - 1 + 7;

                        int index = (planeIndex * 10 * 9) + (r * 9) + c;
                        data[index] = 1.0f;
                    }
                }

                // 1. 创建包含所有 data 的 Tensor
                using var fullTensor = torch.tensor(data, dtype: ScalarType.Float32);

                // 2. 截取并变形
                using var narrowed = fullTensor.narrow(0, 0, DataSize);
                using var view3D = narrowed.reshape(14, 10, 9);

                // 3. 翻转视角 (使用静态常量 FlipDims，避免 new 带来 GC)
                using var processedView = isRedTurn ? view3D.alias() : view3D.flip(FlipDims);

                // 4. 增加 Batch 维度并立即克隆
                using var finalView4D = processedView.unsqueeze(0);

                return finalView4D.clone();
            }
            finally
            {
                // 3. 归还数组
                ArrayPool<float>.Shared.Return(data);
            }
        }
    }
}
