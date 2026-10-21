import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def check_nii_file(filepath):
    print(f"\n[INFO] 正在检查文件: {filepath}")

    if not os.path.isfile(filepath):
        print("[ERROR] 文件不存在。")
        return None

    try:
        image = sitk.ReadImage(filepath)
        print("[OK] 图像读取成功。")
    except Exception as e:
        print(f"[ERROR] 图像读取失败: {e}")
        return None

    size = image.GetSize()
    if any(s == 0 for s in size):
        print("[ERROR] 图像尺寸为0，文件可能为空或损坏。")
        return None

    print("图像尺寸 (Size):", size)
    print("图像分辨率 (Spacing):", image.GetSpacing())
    print("图像起点 (Origin):", image.GetOrigin())
    print("图像方向 (Direction):", image.GetDirection())

    return image


def visualize_middle_slice(image, title=""):
    array = sitk.GetArrayFromImage(image)  # z, y, x
    z, y, x = array.shape

    mid_z = z // 2
    mid_y = y // 2
    mid_x = x // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(array[mid_z, :, :], cmap='gray')
    axs[0].set_title(f'Axial (Z={mid_z})')

    axs[1].imshow(array[:, mid_y, :], cmap='gray')
    axs[1].set_title(f'Coronal (Y={mid_y})')

    axs[2].imshow(array[:, :, mid_x], cmap='gray')
    axs[2].set_title(f'Sagittal (X={mid_x})')

    for ax in axs:
        ax.axis('off')

    plt.suptitle(f"中间切片 - {title}")
    plt.tight_layout()
    plt.show()


def scan_folder(folder_path, visualize=False):
    print(f"[INFO] 开始扫描文件夹: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                fpath = os.path.join(root, fname)
                image = check_nii_file(fpath)
                if image and visualize:
                    visualize_middle_slice(image, title=fname)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量检查MRI图像并可选可视化")
    parser.add_argument("folder", type=str, help="包含.nii图像的文件夹路径")
    parser.add_argument("--visualize", action="store_true", help="是否显示图像切片")
    args = parser.parse_args()

    scan_folder(args.folder, visualize=args.visualize)
