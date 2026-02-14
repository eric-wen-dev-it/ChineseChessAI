using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    public class ResBlock : Module<Tensor, Tensor>
    {
        // 明确类型，避免抽象类导致的梯度丢失
        private Conv2d conv1;
        private BatchNorm2d bn1;
        private Conv2d conv2;
        private BatchNorm2d bn2;

        public ResBlock(long channels) : base("ResBlock")
        {
            conv1 = Conv2d(channels, channels, 3, 1, 1, bias: false);
            bn1 = BatchNorm2d(channels);
            conv2 = Conv2d(channels, channels, 3, 1, 1, bias: false);
            bn2 = BatchNorm2d(channels);

            // 显式逐个注册
            register_module("conv1", conv1);
            register_module("bn1", bn1);
            register_module("conv2", conv2);
            register_module("bn2", bn2);
        }

        public override Tensor forward(Tensor x)
        {
            // 严禁在 forward 内部使用 using 或 Dispose
            var h1 = conv1.forward(x);
            var h2 = bn1.forward(h1);
            var h3 = torch.nn.functional.relu(h2);

            var h4 = conv2.forward(h3);
            var h5 = bn2.forward(h4);

            // 残差连接
            var sum = h5.add(x);
            return torch.nn.functional.relu(sum);
        }
    }
}