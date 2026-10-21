import os
import sys
import subprocess
import shutil

def find_input_images(input_dir):
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')]

def main():
    input_dir = os.getenv('INPUT_IMAGES_DIR', '/input/images')
    output_dir = os.getenv('OUTPUT_MASKS_DIR', '/output')
    os.makedirs(output_dir, exist_ok=True)

    cases = find_input_images(input_dir)
    for case in cases:
        cmd = [
            'nnUNetv2_predict',
            '-i', case,
            '-o', output_dir,
            '-d', '101',
            '-c', '3d_fullres',
            '-tr', 'nnUNetTrainerV2',
            '-chk', 'model_best'  # 建议使用验证表现最好的模型
        ]
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
