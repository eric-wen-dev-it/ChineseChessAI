using ChineseChessAI.Core;
using System;
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
                using var view3D = fullTensor.narrow(0, 0, DataSize).reshape(14, 10, 9);

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