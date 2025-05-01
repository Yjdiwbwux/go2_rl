import torch
import torch.nn as nn
from torch.onnx import export

class ActorMLP(nn.Module):
    def __init__(self):
        super(ActorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1160, 512),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 12)
        )
    
    def forward(self, x):
        return self.model(x)

def convert_actor_to_onnx(model_path="/home/panda/go2_rl_crouch/go2_rl/logs/rough_go2/exported/policies/policy_1.pt", onnx_path="/home/panda/go2_rl_crouch/go2_rl/logs/rough_go2/exported/policies/model.onnx"):
    # 1. 创建或加载模型
    model = ActorMLP()  # 如果是新模型
    # 或者加载预训练权重:
    # model.load_state_dict(torch.load(model_path))
    
    # 2. 设置为评估模式
    model.eval()
    
    # 3. 创建虚拟输入 (batch_size=1, input_dim=1160)
    dummy_input = torch.randn(1, 1160)
    
    # 4. 导出为ONNX
    export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,  # 使用较新的opset以支持ELU等算子
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # 支持动态批量大小
            'output': {0: 'batch_size'}
        }
    )
    print(f"Actor MLP模型已成功导出到 {onnx_path}")

# 验证ONNX模型
def verify_onnx_model(onnx_path="/home/panda/go2_rl_crouch/go2_rl/logs/rough_go2/exported/policies/model.onnx"):
    import onnxruntime as ort
    
    # 创建推理会话
    ort_session = ort.InferenceSession(onnx_path)
    
    # 准备测试输入
    test_input = torch.randn(1, 1160).numpy()
    
    # 运行推理
    outputs = ort_session.run(None, {'input': test_input})
    print("ONNX模型输出形状:", outputs[0].shape)  # 应为(1, 12)

if __name__ == "__main__":
    convert_actor_to_onnx()
    verify_onnx_model()