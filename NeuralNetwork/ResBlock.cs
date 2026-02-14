using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class ResBlock : Module<Tensor, Tensor>
{
    private Module<Tensor, Tensor> conv1;
    private Module<Tensor, Tensor> bn1;
    private Module<Tensor, Tensor> conv2;
    private Module<Tensor, Tensor> bn2;

    public ResBlock(long channels) : base("ResBlock")
    {
        // 1. 初始化层
        conv1 = Conv2d(channels, channels, 3, 1, 1, bias: false);
        bn1 = BatchNorm2d(channels);
        conv2 = Conv2d(channels, channels, 3, 1, 1, bias: false);
        bn2 = BatchNorm2d(channels);

        // 2. 显式注册每一个模块，不要同时调用 RegisterComponents()
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
    }

    public override Tensor forward(Tensor x)
    {
        // 3. 使用 functional.relu，这比重复使用同一个 relu 模块更安全
        var h = torch.nn.functional.relu(bn1.forward(conv1.forward(x)));
        var outTensor = bn2.forward(conv2.forward(h));

        // 4. 残差相加
        var residual = outTensor.add(x);
        return torch.nn.functional.relu(residual);
    }
}