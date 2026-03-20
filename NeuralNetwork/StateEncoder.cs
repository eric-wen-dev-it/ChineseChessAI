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

        public static Tensor Encode(Board board)
        {
            // 1. 从内存池租借数组（Rent 返回的数组长度通常 >= DataSize，例如 2048）
            float[] data = ArrayPool<float>.Shared.Rent(DataSize);

            try
            {
                // 2. 清空数组（租来的可能有旧数据）
                Array.Clear(data, 0, DataSize);

                bool isRedTurn = board.IsRedTurn;

                // --- 填值逻辑 (保持不变) ---
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

                // 【终极防漏水修复】：必须用 using 包裹创建出的每一个中间 Tensor！
                // 1. 创建包含所有 data 的 Tensor (可能长度是 2048)
                using var fullTensor = torch.tensor(data, dtype: ScalarType.Float32);

                // 2. 截取前 1260 个数据并变形 (这是一个 View 视图，依赖 fullTensor)
                using var view3D = fullTensor.narrow(0, 0, DataSize).reshape(14, 10, 9);

                // 3. 翻转视角 (flip 返回新 Tensor，alias 返回原 Tensor 的替身)
                using var processedView = isRedTurn ? view3D.alias() : view3D.flip(new long[] { 1, 2 });

                // 4. 增加 Batch 维度，并立刻执行 .clone() ！
                using var finalView4D = processedView.unsqueeze(0);

                // 【最关键的一步】：clone() 会在底层开辟一块全新的、干净的内存块。
                // 这样当方法结束，所有的 using 变量被销毁时，只有这个 clone 出来的结果存活并返回。
                return finalView4D.clone();
            }
            finally
            {
                // 3. 必须归还数组，否则会导致 C# 端的内存泄漏
                ArrayPool<float>.Shared.Return(data);
            }
        }
    }
}