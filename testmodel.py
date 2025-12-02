# 用于快速验证 lightglue_tssa 模型能否在本仓库中跑通的最小测试脚本

import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

# 将工程根目录加入 sys.path，确保能 import gluefactory 包
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载配置文件
cfg_path = PROJECT_ROOT / "gluefactory" / "configs" / "superpoint+lightglue_megadepth_v1.yaml"
cfg = OmegaConf.load(str(cfg_path))

# 直接使用实现文件中的 LightGlue
from gluefactory.models.matchers.lightglue_tssa_efficient import LightGlue

def make_dummy_batch(
    batch_size=1, m=64, n=72, desc_dim=256, image_size=(640, 480), device="cpu"
):
    B = batch_size
    keypoints0 = torch.rand(B, m, 2, device=device) * torch.tensor(image_size[::-1], device=device)
    keypoints1 = torch.rand(B, n, 2, device=device) * torch.tensor(image_size[::-1], device=device)
    descriptors0 = torch.randn(B, m, desc_dim, device=device)
    descriptors1 = torch.randn(B, n, desc_dim, device=device)

    data = {
        "keypoints0": keypoints0,
        "keypoints1": keypoints1,
        "descriptors0": descriptors0,
        "descriptors1": descriptors1,
        "view0": {"image_size": torch.tensor(image_size[::-1], device=device)},
        "view1": {"image_size": torch.tensor(image_size[::-1], device=device)},
    }
    return data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 使用配置里的 matcher 部分作为 conf（会与模型默认配置合并）
    matcher_conf = OmegaConf.to_container(cfg.model.matcher, resolve=True)
    model = LightGlue(matcher_conf)
    model.eval()
    model.to(device)

    # desc_dim 与模型默认 input_dim 一致（LightGlue.default_conf 默认 256）
    data = make_dummy_batch(batch_size=1, m=128, n=140, desc_dim=model.conf.input_dim, image_size=(1024, 768), device=device)

    with torch.no_grad():
        out = model(data)

    # 简要打印结果以确认前向通过
    print("模型前向成功，输出键：", list(out.keys()))
    print("matches0 shape:", out["matches0"].shape)
    print("matching_scores0 shape:", out["matching_scores0"].shape)
    # 若需要可查看第一个匹配对示例
    print("matches0[0,:10]:", out["matches0"][0, :10].cpu().numpy())

if __name__ == "__main__":
    main()