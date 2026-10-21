# eval_segmentation.py

import os
import json
import SimpleITK as sitk
import numpy as np
from typing import List

# 假设你提供了这几个函数，可以替换为你自己的实现
from your_metrics import dice, jaccard, hausdorff94  # 示例名

def load_image(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

def eval_pair(pred_path: str, gt_path: str) -> dict:
    pred = load_image(pred_path)
    gt = load_image(gt_path)
    stats = {
        "dice": float(dice(pred, gt)),
        "jaccard": float(jaccard(pred, gt)),
        "hausdorff94": float(hausdorff94(pred, gt)),
    }
    return stats

def evaluate_folder(pred_dir: str, gt_dir: str, out_json: str):
    results = {}
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith('.nii.gz'):
            continue
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found: {gt_path}")
            continue
        stats = eval_pair(pred_path, gt_path)
        results[fname] = stats
        print(f"{fname}: {stats}")

    # 计算平均值
    avg = {k: np.mean([v[k] for v in results.values()]) for k in ("dice", "jaccard", "hausdorff94")}
    summary = {"average": avg, "per_image": results}

    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument("--pred", required=True, help="预测图像目录 (*.nii.gz)")
    parser.add_argument("--gt", required=True, help="True label 目录 (*.nii.gz)")
    parser.add_argument("--out", default="eval_summary.json", help="输出 JSON 文件")
    args = parser.parse_args()

    evaluate_folder(args.pred, args.gt, args.out)
