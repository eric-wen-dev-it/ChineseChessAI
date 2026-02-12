using System;
using TorchSharp;
using ChineseChessAI.Core;
using static TorchSharp.torch;

namespace ChineseChessAI.NeuralNetwork
{
    public static class StateEncoder
    {
        public static Tensor Encode(Board board)
        {
            float[] data = new float[14 * 10 * 9];
            bool isRedTurn = board.IsRedTurn;

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

            // 修复点：显式使用 torch.tensor 避免与变量名冲突
            // 或者将变量名改为 encodedTensor
            var encodedTensor = torch.tensor(data, new long[] { 14, 10, 9 });

            if (!isRedTurn)
            {
                encodedTensor = FlipBoard(encodedTensor);
            }

            return encodedTensor.unsqueeze(0);
        }

        private static Tensor FlipBoard(Tensor t)
        {
            // 沿行(dim 1)和列(dim 2)进行翻转
            return t.flip(new long[] { 1, 2 });
        }
    }
}