using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    /// <summary>
    /// 基于 ResNet 的中国象棋双头神经网络
    /// </summary>
    public class CChessNet : Module<Tensor, (Tensor, Tensor)>
    {
        private readonly Module<Tensor, Tensor> convBlock;
        private readonly ModuleList<ResBlock> resBlocks;
        private readonly Module<Tensor, Tensor> policyHead;
        private readonly Module<Tensor, Tensor> valueHead;

        public CChessNet(int numResBlocks = 10, int numFilters = 256) : base("CChessNet")
        {
            // 参数含义：(输入通道, 输出通道, 卷积核大小, 步长, 填充)
            convBlock = Sequential(
                Conv2d(14, numFilters, 3, 1, 1, bias: false),
                BatchNorm2d(numFilters),
                ReLU()
            );

            resBlocks = new ModuleList<ResBlock>();
            for (int i = 0; i < numResBlocks; i++)
            {
                resBlocks.Add(new ResBlock(numFilters));
            }

            // 策略头：输出 2 个平面
            
            policyHead = Sequential(
                Conv2d(numFilters, 2, 1, 1, 0, bias: false),
                BatchNorm2d(2),
                ReLU(),
                Flatten(),
                Linear(2 * 10 * 9, 8100) // 2086 修改为 8100
            );

            // 价值头：输出 1 个平面
            valueHead = Sequential(
                Conv2d(numFilters, 1, 1, 1, 0, bias: false),
                BatchNorm2d(1),
                ReLU(),
                Flatten(),
                Linear(1 * 10 * 9, 256),
                ReLU(),
                Linear(256, 1),
                Tanh()
            );

            RegisterComponents();

            if (cuda.is_available())
                this.to(DeviceType.CUDA);
        }

        public override (Tensor, Tensor) forward(Tensor x)
        {
            // 确保输入在正确的设备上
            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;
            x = x.to(device);

            // 基础特征提取
            var outTensor = convBlock.forward(x);

            // 通过残差层
            foreach (var block in resBlocks)
            {
                outTensor = block.forward(outTensor);
            }

            // 返回双头输出
            var policy = policyHead.forward(outTensor);
            var value = valueHead.forward(outTensor);

            return (policy, value);
        }
    }
}