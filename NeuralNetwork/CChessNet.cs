using System;
using System.Collections.Generic; // 必须引入
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    public class CChessNet : Module<Tensor, (Tensor, Tensor)>
    {
        private Module<Tensor, Tensor> convBlock;
        // 核心修复：显式定义残差块数组，方便 RegisterComponents 扫描
        private List<ResBlock> resBlockList;
        private Module<Tensor, Tensor> policyHead;
        private Module<Tensor, Tensor> valueHead;

        public CChessNet(int numResBlocks = 10, int numFilters = 128) : base("CChessNet")
        {
            convBlock = Sequential(
                Conv2d(14, numFilters, 3, 1, 1, bias: false),
                BatchNorm2d(numFilters),
                ReLU()
            );

            resBlockList = new List<ResBlock>();
            for (int i = 0; i < numResBlocks; i++)
            {
                var block = new ResBlock(numFilters);
                resBlockList.Add(block);
                // 【关键步】必须给每个 ResBlock 显式起名并注册，否则计算图必断
                register_module($"resBlock_{i}", block);
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

            register_module("convBlock", convBlock);
            register_module("policyHead", policyHead);
            register_module("valueHead", valueHead);

            RegisterComponents(); // 确保所有通过 register_module 挂载的子模块被激活

            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;
            this.to(device);
        }

        public override (Tensor, Tensor) forward(Tensor x)
        {
            // 确保输入设备一致
            var currentDevice = this.convBlock.parameters().First().device;
            if (x.device.type != currentDevice.type)
                x = x.to(currentDevice);

            // 执行计算
            using (var out1 = convBlock.forward(x))
            {
                var current = out1;
                // 核心修复：依次通过残差块，必须保留张量引用
                for (int i = 0; i < resBlockList.Count; i++)
                {
                    var next = resBlockList[i].forward(current);
                    // 只有在不是第一层时才尝试 dispose 旧层，但由于我们要保留梯度，
                    // 建议在训练模式下减少手动 dispose 以免切断计算图
                    current = next;
                }

                var policy = policyHead.forward(current);
                var value = valueHead.forward(current);

                return (policy, value);
            }
        }
    }
}