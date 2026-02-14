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
        // 核心：保持计算链的绝对连贯，不要在中间手动 Dispose 任何张量
        // 第一层卷积 -> BN -> ReLU
        var h = relu.forward(bn1.forward(conv1.forward(x)));

        // 第二层卷积 -> BN
        var outTensor = bn2.forward(conv2.forward(h));

        // 残差连接：x (原始输入) + outTensor (处理后的特征)
        // 使用 .add(x) 是正确的，这会建立加法节点的梯度联系
        var residual = outTensor.add(x);

        // 最后一次激活并返回
        return relu.forward(residual);
    }
}