using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    public class CChessNet : Module<Tensor, (Tensor, Tensor)>
    {
        private readonly Conv2d conv1;
        private readonly BatchNorm2d bn1;
        private readonly Module<Tensor, Tensor> relu;
        private readonly Module<Tensor, Tensor> resBlocks;
        private readonly Module<Tensor, Tensor> policyHead;
        private readonly Module<Tensor, Tensor> valueHead;

        public CChessNet(int numResBlocks = 20, int numFilters = 192, bool autoCuda = true) : base("CChessNet")
        {
            // 1. 输入层
            // 【修复核心】：不再使用命名参数(kernelSize等)，改用全位置参数，彻底杜绝参数名报错。
            // 参数顺序: in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode, groups, bias
            // 我们显式填入所有参数，并用 'L' 强制指定为 long 类型
            conv1 = Conv2d(14, numFilters, 3L, 1L, 1L, 1L, PaddingModes.Zeros, 1L, false);

            bn1 = BatchNorm2d(numFilters);
            relu = ReLU();

            // 2. 残差塔
            var blocks = new List<Module<Tensor, Tensor>>();
            for (int i = 0; i < numResBlocks; i++)
            {
                blocks.Add(new ResBlock(numFilters));
            }
            resBlocks = Sequential(blocks);

            // 3. 策略头 (Policy Head)
            policyHead = Sequential(
                // 1x1 卷积: kernel=1, stride=1, padding=0, bias=false
                Conv2d(numFilters, 2, 1L, 1L, 0L, 1L, PaddingModes.Zeros, 1L, false),
                BatchNorm2d(2),
                ReLU(),
                Flatten(1L), // 这里的 1L 表示 start_dim
                Linear(2 * 10 * 9, 8100)
            );

            // 4. 价值头 (Value Head)
            valueHead = Sequential(
                // 1x1 卷积: kernel=1, stride=1, padding=0, bias=false
                Conv2d(numFilters, 1, 1L, 1L, 0L, 1L, PaddingModes.Zeros, 1L, false),
                BatchNorm2d(1),
                ReLU(),
                Flatten(1L),
                Linear(1 * 10 * 9, 256),
                ReLU(),
                Linear(256, 1),
                Tanh()
            );

            RegisterComponents();

            if (autoCuda && cuda.is_available())
            {
                this.to(DeviceType.CUDA);
            }
        }

        public override (Tensor, Tensor) forward(Tensor x)
        {
            using var scope = torch.NewDisposeScope();

            using var input = x.device.type != conv1.weight.device.type
                ? x.to(conv1.weight.device)
                : x.alias();

            var out1 = relu.forward(bn1.forward(conv1.forward(input)));
            var towerOut = resBlocks.forward(out1);

            var p = policyHead.forward(towerOut);
            var v = valueHead.forward(towerOut);

            return (p.MoveToOuterDisposeScope(), v.MoveToOuterDisposeScope());
        }
    }
}
