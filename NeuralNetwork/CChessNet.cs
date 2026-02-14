using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules; // 必须引用
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    public class CChessNet : Module<Tensor, (Tensor, Tensor)>
    {
        private Module<Tensor, Tensor> convBlock;
        // 核心修复：必须使用 ModuleList，否则 forward 里的梯度链条无法自动连接
        private ModuleList<ResBlock> resBlocks;
        private Module<Tensor, Tensor> policyHead;
        private Module<Tensor, Tensor> valueHead;

        public CChessNet(int numResBlocks = 10, int numFilters = 128) : base("CChessNet")
        {
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

            policyHead = Sequential(
                Conv2d(numFilters, 2, 1, 1, 0, bias: false),
                BatchNorm2d(2),
                ReLU(),
                Flatten(),
                Linear(2 * 10 * 9, 8100)
            );

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

            // 显式注册所有顶层组件
            register_module("convBlock", convBlock);
            register_module("resBlocks", resBlocks);
            register_module("policyHead", policyHead);
            register_module("valueHead", valueHead);

            RegisterComponents();

            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;
            this.to(device);
        }

        public override (Tensor, Tensor) forward(Tensor x)
        {
            // 1. 确保输入在模型所在的设备上
            var device = this.convBlock.parameters().First().device;
            var input = x.device.type != device.type ? x.to(device) : x;

            // 2. 链式前向计算，严禁使用 using 或中间 Dispose
            var output = convBlock.forward(input);

            // 3. 遍历残差块
            foreach (var block in resBlocks)
            {
                output = block.forward(output);
            }

            // 4. 计算头输出
            var p = policyHead.forward(output);
            var v = valueHead.forward(output);

            return (p, v);
        }

    }
}