import json
import os

def fix_dataset_json():
    # 读取原始dataset.json
    input_file = 'D:/Project/MAMA_MIA/code/nnUNet_raw/Dataset101_BreastMRI/dataset.json'
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 创建新的符合nnUNetv2要求的数据结构
    new_data = {
        "channel_names": {
            "0": "DCE_Phase0",
            "1": "DCE_Phase1",
            "2": "DCE_Phase2",
            "3": "DCE_Phase3",
            "4": "DCE_Phase4"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": data["numTraining"],
        "file_ending": data["file_ending"]
    }
    
    # 保存新的dataset.json
    output_file = 'D:/Project/MAMA_MIA/code/nnUNet_raw/Dataset101_BreastMRI/dataset.json'
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print(f"已创建新的dataset.json文件：{output_file}")

if __name__ == "__main__":
    fix_dataset_json()