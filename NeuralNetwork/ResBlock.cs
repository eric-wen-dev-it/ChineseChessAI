using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class ResBlock : Module<Tensor, Tensor>
{
    private Module<Tensor, Tensor> conv1;
    private Module<Tensor, Tensor> bn1;
    private Module<Tensor, Tensor> conv2;
    private Module<Tensor, Tensor> bn2;
    private Module<Tensor, Tensor> relu;

    public ResBlock(long channels) : base("ResBlock")
    {
        conv1 = Conv2d(channels, channels, 3, 1, 1, bias: false);
        bn1 = BatchNorm2d(channels);
        conv2 = Conv2d(channels, channels, 3, 1, 1, bias: false);
        bn2 = BatchNorm2d(channels);
        relu = ReLU();

        // 【关键步】必须显式注册，否则其参数不会被 CChessNet 识别
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // 核心修复：直接使用算子，不要进行任何可能断开梯度的临时转换
        var outTensor = relu.forward(bn1.forward(conv1.forward(x)));
        outTensor = bn2.forward(conv2.forward(outTensor));

        // 必须通过 add 方法合并，并再次 ReLU
        return relu.forward(outTensor.add(x));
    }
}