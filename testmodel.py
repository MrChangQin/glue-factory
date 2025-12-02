import sys
from pathlib import Path
import torch

# 确保能 import 项目模块（从 repo 根）
repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from gluefactory.models.matchers.lightglue_tssa import LightGlue

def make_dummy(batch=1, m=8, n=10, dim=64, device="cpu"):
    kpts0 = torch.rand(batch, m, 2, device=device)
    kpts1 = torch.rand(batch, n, 2, device=device)
    desc0 = torch.rand(batch, m, dim, device=device)
    desc1 = torch.rand(batch, n, dim, device=device)
    data = {
        "keypoints0": kpts0,
        "keypoints1": kpts1,
        "descriptors0": desc0,
        "descriptors1": desc1,
        "view0": {"image_size": (128, 128)},
        "view1": {"image_size": (128, 128)},
    }
    return data

def test_forward_and_train(device="cpu"):
    conf = {
        "n_layers": 2,
        "num_heads": 2,
        "input_dim": 64,
        "descriptor_dim": 64,
        "flash": False,
        "mp": False,
        "width_confidence": -1,
        "depth_confidence": -1,
        "filter_threshold": 0.0,
        "weights": None,
    }

    model = LightGlue(conf).to(device)

    # 推理测试
    model.eval()
    data = make_dummy(device=device)
    with torch.no_grad():
        pred = model(data)
    print("inference keys:", list(pred.keys()))
    assert "matches0" in pred and "matching_scores0" in pred

    # 训练测试（单步）
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    data = make_dummy(device=device)
    pred = model(data)
    losses, metrics = model.loss(pred, data)
    loss = losses["total"]
    loss.backward()
    opt.step()

    # 检查有梯度（至少一个参数）
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "no gradients after backward"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("running on", device)
    test_forward_and_train(device=device)
    print("OK")