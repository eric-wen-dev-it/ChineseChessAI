using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    /// <summary>
    /// 基于 ResNet 的中国象棋双头神经网络 (优化注册版)
    /// </summary>
    public class CChessNet : Module<Tensor, (Tensor, Tensor)>
    {
        // 组件声明
        private readonly Module<Tensor, Tensor> convBlock;
        private readonly ModuleList<ResBlock> resBlocks;
        private readonly Module<Tensor, Tensor> policyHead;
        private readonly Module<Tensor, Tensor> valueHead;

        public CChessNet(int numResBlocks = 10, int numFilters = 256) : base("CChessNet")
        {
            // 1. 卷积块：特征提取
            convBlock = Sequential(
                Conv2d(14, numFilters, 3, 1, 1, bias: false),
                BatchNorm2d(numFilters),
                ReLU()
            );

            // 2. 残差块组
            resBlocks = new ModuleList<ResBlock>();
            for (int i = 0; i < numResBlocks; i++)
            {
                resBlocks.Add(new ResBlock(numFilters));
            }

            // 3. 策略头 (Policy Head)：输出所有可能走法的概率分布 (8100维)
            policyHead = Sequential(
                Conv2d(numFilters, 2, 1, 1, 0, bias: false),
                BatchNorm2d(2),
                ReLU(),
                Flatten(),
                Linear(2 * 10 * 9, 8100)
            );

            // 4. 价值头 (Value Head)：输出当前局面的胜率评估 (-1 到 1)
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

            // 【修复核心】将 RegisterModule 改为 register_module
            register_module("convBlock", convBlock);
            register_module("resBlocks", resBlocks);
            register_module("policyHead", policyHead);
            register_module("valueHead", valueHead);

            // 这一行通常会自动扫描并注册类中的所有 Module 字段
            RegisterComponents();

            if (cuda.is_available())
                this.to(DeviceType.CUDA);
        }

        /// <summary>
        /// 前向传播逻辑
        /// </summary>
        /// <param name="x">输入张量 [Batch, 14, 10, 9]</param>
        /// <returns>(策略对数概率, 价值预测)</returns>
        public override (Tensor, Tensor) forward(Tensor x)
        {
            // 确保输入张量与模型处于同一设备
            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;
            if (x.device.type != device)
            {
                x = x.to(device);
            }

            // A. 通过基础卷积块提取特征
            var outTensor = convBlock.forward(x);

            // B. 通过残差层链条
            foreach (var block in resBlocks)
            {
                outTensor = block.forward(outTensor);
            }

            // C. 分别计算策略分布和价值评估
            var policy = policyHead.forward(outTensor);
            var value = valueHead.forward(outTensor);

            return (policy, value);
        }
    }
}