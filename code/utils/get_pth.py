import torch

# # 加载模型文件
# checkpoint = torch.load('D:\Project\MAMA_MIA\full_image_dce_mri_tumor_segmentation\full_image_dce_mri_tumor_segmentation\fold_4\checkpoint_final.pth')
#
# # 查看文件内容
# print("文件包含的键:", checkpoint.keys())
# # 通常会输出:
# # ['model_state_dict', 'optimizer_state_dict', 'epoch', 'best_acc']
#
# # 查看具体参数
# print("训练轮次:", checkpoint['epoch'])
# print("最佳准确率:", checkpoint['best_acc'])
#
# # 2. 查看模型参数
# for name, param in checkpoint['model_state_dict'].items():
#     print(f"层名称: {name}")
#     print(f"参数形状: {param.shape}")
#     print(f"参数类型: {param.dtype}")
#     print("---")


pthfile = r'D:\Project\MAMA_MIA\full_image_dce_mri_tumor_segmentation\full_image_dce_mri_tumor_segmentation\fold_4\checkpoint_final.pth'  # .pth文件的路径
# 添加numpy.core.multiarray.scalar到安全全局变量
import torch.serialization
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# 安全加载模型
model = torch.load(pthfile, map_location=torch.device('cpu'), weights_only=True)  # 设置在cpu环境下查询
print('type:')
print(type(model))  # 查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  # 查看模型字典里面的key
    print(k)
print('value:')
for k in model:  # 查看模型字典里面的value
    print(k, model[k])