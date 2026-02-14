using TorchSharp;
using TorchSharp.Modules; // 显式引入以确保 Module 基类识别
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

        // 【最稳妥的注册方式】显式注册每一个层，确保它们被父模块 CChessNet 的 parameters() 捕获
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("relu", relu);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // 保持计算链连贯，直接返回计算结果
        var h = conv1.forward(x);
        h = bn1.forward(h);
        h = relu.forward(h);
        h = conv2.forward(h);
        h = bn2.forward(h);

        // 残差相加并激活
        var residual = h.add(x);
        return relu.forward(residual);
    }
}