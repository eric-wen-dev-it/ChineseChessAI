using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    /// <summary>
    /// 基于 ResNet 的中国象棋双头神经网络
    /// 修复了参数注册失效导致的梯度丢失问题
    /// </summary>
    public class CChessNet : Module<Tensor, (Tensor, Tensor)>
    {
        // 核心：移除 readonly，确保 RegisterComponents 或 register_module 能生效
        private Module<Tensor, Tensor> convBlock;
        private ModuleList<ResBlock> resBlocks;
        private Module<Tensor, Tensor> policyHead;
        private Module<Tensor, Tensor> valueHead;

        public CChessNet(int numResBlocks = 10, int numFilters = 128) : base("CChessNet")
        {
            // 1. 卷积块定义
            convBlock = Sequential(
                Conv2d(14, numFilters, 3, 1, 1, bias: false),
                BatchNorm2d(numFilters),
                ReLU()
            );

            // 2. 残差块组定义
            resBlocks = new ModuleList<ResBlock>();
            for (int i = 0; i < numResBlocks; i++)
            {
                // 注意：ResBlock 内部也必须包含 RegisterComponents 或显式注册
                resBlocks.Add(new ResBlock(numFilters));
            }

            // 3. 策略头定义
            policyHead = Sequential(
                Conv2d(numFilters, 2, 1, 1, 0, bias: false),
                BatchNorm2d(2),
                ReLU(),
                Flatten(),
                Linear(2 * 10 * 9, 8100)
            );

            // 4. 价值头定义
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

            // 【核心修复步骤】
            // 在某些 TorchSharp 版本中，RegisterComponents 对 private 字段支持不佳
            // 必须显式调用 register_module 建立父子模块的参数关联
            register_module("convBlock", convBlock);
            register_module("resBlocks", resBlocks);
            register_module("policyHead", policyHead);
            register_module("valueHead", valueHead);

            // 最后调用这个确保所有通过 register_module 挂载的参数都被提取
            RegisterComponents();

            // 移动到设备
            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;
            this.to(device);
        }

        public override (Tensor, Tensor) forward(Tensor x)
        {
            // 自动处理输入设备的转换
            if (x.device.type != this.convBlock.parameters().First().device.type)
            {
                x = x.to(this.convBlock.parameters().First().device);
            }

            var outTensor = convBlock.forward(x);

            foreach (var block in resBlocks)
            {
                outTensor = block.forward(outTensor);
            }

            var policy = policyHead.forward(outTensor);
            var value = valueHead.forward(outTensor);

            return (policy, value);
        }
    }
}