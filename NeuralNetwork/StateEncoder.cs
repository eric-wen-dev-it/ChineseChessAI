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

                // 【核心修复】
                // 1. 直接传入 float[] 数组，TorchSharp 100% 支持。
                //    注意：此时创建的 Tensor 长度等于 data.Length (可能 > 1260)
                var fullTensor = torch.tensor(data, dtype: ScalarType.Float32);

                // 2. 使用 .narrow(维, 开始, 长度) 截取我们需要的 1260 个有效数据
                // 3. 使用 .reshape 变为 [14, 10, 9]
                var encodedTensor = fullTensor.narrow(0, 0, DataSize).reshape(14, 10, 9);

                // 如果是黑方，翻转视角
                if (!isRedTurn)
                {
                    encodedTensor = FlipBoard(encodedTensor);
                }

                // 增加 Batch 维度: [1, 14, 10, 9]
                return encodedTensor.unsqueeze(0);
            }
            finally
            {
                // 3. 必须归还数组，否则会导致内存泄漏
                ArrayPool<float>.Shared.Return(data);
            }
        }

        private static Tensor FlipBoard(Tensor t)
        {
            return t.flip(new long[] { 1, 2 });
        }
    }
}