using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{
    public class ResBlock : Module<Tensor, Tensor>
    {
        private readonly Module<Tensor, Tensor> conv1;
        private readonly Module<Tensor, Tensor> bn1;
        private readonly Module<Tensor, Tensor> conv2;
        private readonly Module<Tensor, Tensor> bn2;
        private readonly Module<Tensor, Tensor> relu;

        public ResBlock(long channels) : base("ResBlock")
        {
            // 注意：在 TorchSharp 中，参数顺序为 (inputChannels, outputChannels, kernelSize, stride, padding)
            // bias: false 是因为后面紧跟 BatchNorm，偏置会被抵消
            conv1 = Conv2d(channels, channels, 3, 1, 1, bias: false);
            bn1 = BatchNorm2d(channels);

            conv2 = Conv2d(channels, channels, 3, 1, 1, bias: false);
            bn2 = BatchNorm2d(channels);

            relu = ReLU();

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var identity = x; // 记忆输入以进行残差连接

            var outTensor = relu.forward(bn1.forward(conv1.forward(x)));
            outTensor = bn2.forward(conv2.forward(outTensor));

            outTensor = outTensor.add(identity); // 核心：残差相加
            return relu.forward(outTensor);
        }
    }
}