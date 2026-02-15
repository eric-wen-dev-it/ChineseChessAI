using TorchSharp;
using TorchSharp.Modules; // 必须引用
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ChineseChessAI.NeuralNetwork
{

    public class CChessNet : Module<Tensor, (Tensor, Tensor)>
    {
        // 拆解 Sequential，直接管理层级
        private Conv2d conv1;
        private BatchNorm2d bn1;
        private List<ResBlock> resBlockList;

        // 策略头拆解
        private Conv2d p_conv;
        private BatchNorm2d p_bn;
        private Linear p_fc;

        // 价值头拆解
        private Conv2d v_conv;
        private BatchNorm2d v_bn;
        private Linear v_fc1;
        private Linear v_fc2;

        public CChessNet(int numResBlocks = 10, int numFilters = 128) : base("CChessNet")
        {
            conv1 = Conv2d(14, numFilters, 3, 1, 1, bias: false);
            bn1 = BatchNorm2d(numFilters);
            register_module("conv1", conv1);
            register_module("bn1", bn1);

            resBlockList = new List<ResBlock>();
            for (int i = 0; i < numResBlocks; i++)
            {
                var block = new ResBlock(numFilters);
                resBlockList.Add(block);
                register_module($"resBlock_{i}", block); // 关键：手动逐个注册
            }

            p_conv = Conv2d(numFilters, 2, 1, 1, 0, bias: false);
            p_bn = BatchNorm2d(2);
            p_fc = Linear(2 * 10 * 9, 8100);
            register_module("p_conv", p_conv);
            register_module("p_bn", p_bn);
            register_module("p_fc", p_fc);

            v_conv = Conv2d(numFilters, 1, 1, 1, 0, bias: false);
            v_bn = BatchNorm2d(1);
            v_fc1 = Linear(1 * 10 * 9, 256);
            v_fc2 = Linear(256, 1);
            register_module("v_conv", v_conv);
            register_module("v_bn", v_bn);
            register_module("v_fc1", v_fc1);
            register_module("v_fc2", v_fc2);

            var device = cuda.is_available() ? DeviceType.CUDA : DeviceType.CPU;
            this.to(device);
        }

        public override (Tensor, Tensor) forward(Tensor x)
        {
            var device = conv1.weight.device;
            var input = x.device.type != device.type ? x.to(device) : x;

            // 基础特征提取
            var h = torch.nn.functional.relu(bn1.forward(conv1.forward(input)));

            foreach (var block in resBlockList)
            {
                h = block.forward(h);
            }

            // 策略分支
            var ph1 = torch.nn.functional.relu(p_bn.forward(p_conv.forward(h)));
            // ✅ 修复：必须使用 flatten(1) 来保留 Batch 维度
            var ph2 = ph1.flatten(1);
            var pLogits = p_fc.forward(ph2);

            // 价值分支
            var vh1 = torch.nn.functional.relu(v_bn.forward(v_conv.forward(h)));
            // ✅ 修复：这里也要改为 flatten(1)
            var vh2 = vh1.flatten(1);
            var vh3 = torch.nn.functional.relu(v_fc1.forward(vh2));
            var vPred = torch.tanh(v_fc2.forward(vh3));

            return (pLogits, vPred);
        }
    }

}
