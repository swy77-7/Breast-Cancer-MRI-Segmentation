#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理脚本 - 用于MAMA-MIA数据集中DCE-MRI影像的预处理
该脚本实现：
1. z-score归一化：对所有对比相（从预对比到最后一个对比后）使用相同的均值和标准差进行z-score归一化
2. 等向素重采样：将MRI重采样为均匀像素间距[1,1,1]
3. 可选的偏置场校正：使用N4算法校正MRI中的偏置场
4. 可选的非局部均值降噪：减少图像噪声
"""

import os
import re
import glob
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import time
from skimage import restoration
import sys

def setup_logger(log_file=None, log_level=logging.INFO):
    """设置日志记录器

    参数:
    log_file: 日志文件路径，如果为None则只输出到控制台
    log_level: 日志级别

    返回:
    logger对象
    """
    logger = logging.getLogger("MAMA-MIA-Preprocessing")
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 创建全局日志记录器
logger = setup_logger()

def read_mri_phase_from_patient_id(images_folder, patient_id, phase=0):
    '''读取指定病人ID的指定相位的MRI图像
    
    参数:
    images_folder: str - 包含MRI图像的文件夹路径
    patient_id: str - 病人ID
    phase: int - MRI图像的相位 (0: 预对比, 1: 第一个对比后, 2: 第二个对比后, ...)
    
    返回:
    SimpleITK图像对象
    '''
    file_path = os.path.join(images_folder, f"{patient_id}_000{phase}.nii.gz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    phase_sitk = sitk.ReadImage(file_path, sitk.sitkFloat32)
    return phase_sitk

def read_segmentation_from_patient_id(segmentations_folder, patient_id):
    '''读取指定病人ID的分割结果
    
    参数:
    segmentations_folder: str - 包含肿瘤分割结果的文件夹路径
    patient_id: str - 病人ID
    
    返回:
    SimpleITK图像对象
    '''
    file_path = os.path.join(segmentations_folder, f"{patient_id}.nii.gz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"分割文件不存在: {file_path}")
    
    mask_sitk = sitk.ReadImage(file_path, sitk.sitkUInt8)
    return mask_sitk

def get_image_orientation_from_direction(img_sitk):
    """获取图像方向
    
    参数:
    img_sitk: SimpleITK图像对象
    
    返回:
    方向字符串
    """
    filter = sitk.DICOMOrientImageFilter()
    filter.Execute(img_sitk)
    orientation = filter.GetOrientationFromDirectionCosines(img_sitk.GetDirection())
    return orientation

def bias_correction_sitk(image_sitk, otsu_threshold=False, shrink_factor=0):
    """应用N4偏置场校正
    
    参数:
    image_sitk: SimpleITK图像对象
    otsu_threshold: 是否使用Otsu阈值分割创建掩码
    shrink_factor: 如果>0，将先缩小图像再进行校正，以加速处理
    
    返回:
    校正后的SimpleITK图像对象
    """
    try:
        if shrink_factor:
            # N4BiasFieldCorrectionImageFilter运行时间长，先缩小图像
            mask_breast = sitk.OtsuThreshold(image_sitk, 0, 1)
            shrinked_image_sitk = sitk.Shrink(image_sitk, [shrink_factor] * image_sitk.GetDimension())
            shrinked_mask_breast = sitk.Shrink(mask_breast, [shrink_factor] * mask_breast.GetDimension())
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            tmp_image = corrector.Execute(shrinked_image_sitk, shrinked_mask_breast)
            log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
            corrected_image_sitk = image_sitk / sitk.Exp(log_bias_field)
        else:
            initial_img = image_sitk
            # 转换为浮点型以便进行偏置场校正
            tmp_image = sitk.Cast(image_sitk, sitk.sitkFloat64)
            # 将零值设置为一个很小的数值以防止除零
            tmp_image = sitk.GetArrayFromImage(tmp_image)
            tmp_image[tmp_image == 0] = np.finfo(float).eps
            tmp_image = sitk.GetImageFromArray(tmp_image)
            tmp_image.CopyInformation(initial_img)
            if otsu_threshold:
                maskImage = sitk.OtsuThreshold(tmp_image, 0, 1)
            # 使用N4偏置场校正
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            if otsu_threshold:
                corrected_image_sitk = corrector.Execute(tmp_image, maskImage)
            else:
                corrected_image_sitk = corrector.Execute(tmp_image)
        
        return corrected_image_sitk
    except Exception as e:
        logger.error(f"偏置场校正失败: {str(e)}")
        return image_sitk

def nlmeans_denoise_sitk(image_sitk, patch_size=5, patch_distance=6, h=0.8):
    """使用非局部均值(NLMeans)滤波对DCE-MRI图像进行降噪
    
    参数:
    image_sitk: SimpleITK图像对象
    patch_size: 用于降噪的patch大小
    patch_distance: 搜索patch的最大像素距离
    h: 截止距离（h越大越平滑）
    
    返回:
    降噪后的SimpleITK图像对象
    """
    try:
        image_np = sitk.GetArrayFromImage(image_sitk)
        
        denoised_np = restoration.denoise_nl_means(
            image_np,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            fast_mode=True
        )
        
        denoised_image_sitk = sitk.GetImageFromArray(denoised_np)
        denoised_image_sitk.CopyInformation(image_sitk)
        
        return denoised_image_sitk
    except Exception as e:
        logger.error(f"降噪失败: {str(e)}")
        return image_sitk

def clip_image_sitk(image_sitk, percentiles=[1, 99]):
    """裁剪图像的强度范围到指定百分位数
    
    参数:
    image_sitk: SimpleITK图像对象
    percentiles: 列表，包含两个百分位数值[下限, 上限]
    
    返回:
    裁剪后的SimpleITK图像对象
    """
    try:
        image_array = sitk.GetArrayFromImage(image_sitk)
        image_array = image_array.ravel()
        image_array = image_array[image_array != 0]
        
        if len(image_array) == 0:
            logger.warning("图像中没有非零值，跳过裁剪")
            return image_sitk
            
        lowerbound = np.percentile(image_array, percentiles[0])
        upperbound = np.percentile(image_array, percentiles[1])
        
        filter = sitk.ClampImageFilter()
        filter.SetLowerBound(float(lowerbound))
        filter.SetUpperBound(float(upperbound))

        clipped_image_sitk = filter.Execute(image_sitk)
        return clipped_image_sitk
    except Exception as e:
        logger.error(f"裁剪强度范围失败: {str(e)}")
        return image_sitk

def zscore_normalization_sitk(image_sitk, mean, std):
    """对图像进行Z-score归一化
    
    参数:
    image_sitk: SimpleITK图像对象
    mean: 均值
    std: 标准差
    
    返回:
    归一化后的SimpleITK图像对象
    """
    try:
        array = sitk.GetArrayFromImage(image_sitk)
        
        if std == 0:
            logger.warning("标准差为零，使用1.0代替以避免除以零")
            std = 1.0
            
        normalized_array = (array - mean) / std
        zscored_sitk = sitk.GetImageFromArray(normalized_array)
        zscored_sitk.CopyInformation(image_sitk)
        return zscored_sitk
    except Exception as e:
        logger.error(f"Z-score归一化失败: {str(e)}")
        return image_sitk

def histogram_equalization_sitk(image_sitk, alpha=0.3, beta=0.3, radius=5):
    """对图像进行直方图均衡化
    
    参数:
    image_sitk: SimpleITK图像对象
    alpha: 控制滤波器的行为，从经典直方图均衡化(alpha=0)到非锐化掩蔽(alpha=1)
    beta: 控制滤波器行为，从非锐化掩蔽(beta=0)到透传(beta=1, alpha=1)
    radius: 控制计算局部统计量的区域大小
    
    返回:
    均衡化后的SimpleITK图像对象
    """
    try:
        hqfilter = sitk.AdaptiveHistogramEqualizationImageFilter()
        hqfilter.SetAlpha(alpha)
        hqfilter.SetBeta(beta)
        hqfilter.SetRadius(radius) 
        heq_image = hqfilter.Execute(image_sitk)
        return heq_image
    except Exception as e:
        logger.error(f"直方图均衡化失败: {str(e)}")
        return image_sitk

def resample_sitk(image_sitk, new_spacing=[1, 1, 1], new_size=None, interpolator=sitk.sitkBSpline, tol=0.00001):
    """重采样图像到新的像素间距或大小
    
    参数:
    image_sitk: SimpleITK图像对象
    new_spacing: 新的像素间距，默认为[1, 1, 1]
    new_size: 新的图像大小，如果为None则根据new_spacing计算
    interpolator: 插值方法，默认为B样条插值
    tol: 浮点数比较容差
    
    返回:
    重采样后的SimpleITK图像对象
    """
    try:
        original_size = image_sitk.GetSize()
        original_spacing = image_sitk.GetSpacing()
        
        if len(original_size) == 2:
            original_size = original_size + (1, )
        if len(original_spacing) == 2:
            original_spacing = original_spacing + (1.0, )

        if new_size is None and new_spacing is not None:
            new_size = [
                round(original_size[0] * (original_spacing[0] + tol) / new_spacing[0]),
                round(original_size[1] * (original_spacing[1] + tol) / new_spacing[1]),
                round(original_size[2] * (original_spacing[2] + tol) / new_spacing[2])
            ]
            new_size = np.array(new_size, dtype='int').tolist()

        if new_spacing is None and new_size is not None:
            tol = 0
            new_spacing = [
                original_size[0] * (original_spacing[0] + tol) / new_size[0],
                original_size[1] * (original_spacing[1] + tol) / new_size[1],
                original_size[2] * (original_spacing[2] + tol) / new_size[2]
            ]

        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputDirection(image_sitk.GetDirection())
        resample_filter.SetOutputOrigin(image_sitk.GetOrigin())
        resample_filter.SetOutputPixelType(image_sitk.GetPixelID())
        resample_filter.SetTransform(sitk.Transform())
        
        try:
            resampled_image_sitk = resample_filter.Execute(image_sitk)
        except RuntimeError:
            logger.warning('重采样时输出方向错误，重置方向。')
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            resample_filter.SetOutputDirection(direction)
            image_sitk.SetDirection(direction)
            resampled_image_sitk = resample_filter.Execute(image_sitk)

        return resampled_image_sitk
    except Exception as e:
        logger.error(f"重采样失败: {str(e)}")
        return image_sitk

def preprocess_dce_mri(input_folder, output_folder, patient_id, phases=[0, 1, 2, 3, 4], 
                      do_bias_correction=False, do_denoise=False, do_histogram_eq=False,
                      clip_percentiles=[1, 99], isotropic_spacing=[1, 1, 1]):
    """预处理一个病人的所有DCE-MRI相位"""
    try:
        patient_output_folder = os.path.join(output_folder, patient_id)
        os.makedirs(patient_output_folder, exist_ok=True)
        
        logger.info(f"开始处理病人 {patient_id}")
        
        phase_images = []
        for phase in phases:
            try:
                logger.debug(f"读取病人 {patient_id} 的相位 {phase}")
                img = read_mri_phase_from_patient_id(input_folder, patient_id, phase)
                
                if do_bias_correction:
                    logger.debug(f"对病人 {patient_id} 的相位 {phase} 执行偏置场校正")
                    img = bias_correction_sitk(img, otsu_threshold=True)
                
                if do_denoise:
                    logger.debug(f"对病人 {patient_id} 的相位 {phase} 执行降噪")
                    img = nlmeans_denoise_sitk(img)
                
                logger.debug(f"对病人 {patient_id} 的相位 {phase} 执行强度裁剪")
                img = clip_image_sitk(img, percentiles=clip_percentiles)
                
                if do_histogram_eq:
                    logger.debug(f"对病人 {patient_id} 的相位 {phase} 执行直方图均衡化")
                    img = histogram_equalization_sitk(img)
                
                phase_images.append(img)
            except Exception as e:
                logger.error(f"无法处理病人 {patient_id} 的相位 {phase}: {str(e)}")
                continue

        if not phase_images:
            logger.error(f"病人 {patient_id} 的所有相位都无法处理")
            return False
            
        all_voxels = []
        for img in phase_images:
            array = sitk.GetArrayFromImage(img)
            non_zero_voxels = array[array > 0]
            if len(non_zero_voxels) > 0:
                all_voxels.extend(non_zero_voxels.flatten())
        
        if len(all_voxels) == 0:
            logger.warning(f"警告: 病人 {patient_id} 的所有相位都是空的或只包含零值")
            return False
        
        mean_value = np.mean(all_voxels)
        std_value = np.std(all_voxels)
        logger.info(f"病人 {patient_id} 的所有相位均值: {mean_value:.4f}, 标准差: {std_value:.4f}")
        
        success_count = 0
        for i, phase in enumerate(phases):
            if i < len(phase_images):
                try:
                    logger.debug(f"对病人 {patient_id} 的相位 {phase} 执行等向素重采样")
                    resampled_img = resample_sitk(phase_images[i], new_spacing=isotropic_spacing)
                    
                    logger.debug(f"对病人 {patient_id} 的相位 {phase} 执行Z-score归一化")
                    normalized_img = zscore_normalization_sitk(resampled_img, mean_value, std_value)
                    
                    output_path = os.path.join(patient_output_folder, f"{patient_id}_000{phase}.nii.gz")
                    sitk.WriteImage(normalized_img, output_path)
                    logger.info(f"已保存处理后的图像: {output_path}")
                    success_count += 1
                except Exception as e:
                    logger.error(f"处理病人 {patient_id} 的相位 {phase} 失败: {str(e)}")
        
        logger.info(f"病人 {patient_id} 处理完成，成功处理 {success_count}/{len(phases)} 个相位")
        return success_count > 0
    
    except Exception as e:
        logger.error(f"处理病人 {patient_id} 时发生错误: {str(e)}")
        return False

def preprocess_dce_mri_dataset(input_folder, output_folder, patient_subset=None, **kwargs):
    """预处理整个DCE-MRI数据集
    
    参数:
    input_folder: 输入数据集文件夹路径
    output_folder: 输出数据集文件夹路径
    patient_subset: 如果不为None，只处理此列表中的病人ID
    **kwargs: 传递给preprocess_dce_mri的其他参数
    
    返回:
    处理成功的病人数量
    """
    patient_ids = []
    if patient_subset:
        patient_ids = patient_subset
        logger.info(f"将处理指定的 {len(patient_ids)} 个病人")
    else:
        all_files = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith('.nii.gz'):
                    all_files.append(os.path.join(root, file))
        
        logger.info(f"在目录 {input_folder} 中发现 {len(all_files)} 个NIfTI文件")
        
        if not all_files:
            logger.error("未找到任何.nii.gz文件，请检查输入目录")
            return 0

        patient_ids_set = set()
        for filepath in all_files:
            try:
                filename = os.path.basename(filepath)
                match = re.match(r'^([a-zA-Z]+_\d+)_\d+\.nii\.gz$', filename, re.IGNORECASE)
                if match:
                    patient_id = match.group(1)
                    patient_ids_set.add(patient_id)
                    logger.debug(f"成功解析病人ID: {patient_id} (来自文件: {filename})")
                else:
                    logger.warning(f"跳过文件 {filename} - 文件名格式不符合要求")
            except Exception as e:
                logger.error(f"处理文件 {filename} 时发生错误: {str(e)}")
                continue

        patient_ids = sorted(list(patient_ids_set))
        logger.info(f"共发现 {len(patient_ids)} 个有效病人ID\n示例: {patient_ids[:5]}")

    if not patient_ids:
        logger.error("未找到任何有效的病人ID")
        return 0
    
    os.makedirs(output_folder, exist_ok=True)
    
    success_count = 0
    total_patients = len(patient_ids)
    
    for i, patient_id in enumerate(tqdm(patient_ids, desc="处理病人")):
        logger.info(f"处理病人 {patient_id} ({i+1}/{total_patients})")
        if preprocess_dce_mri(input_folder, output_folder, patient_id, **kwargs):
            success_count += 1
    
    logger.info(f"数据集处理完成，成功处理 {success_count}/{total_patients} 个病人")
    return success_count

def prepare_for_nnunet(preprocessed_folder, nnunet_raw_folder, task_id=101, task_name="BreastMRI"):
    """为nnUNet准备预处理后的数据"""
    try:
        task_folder = os.path.join(nnunet_raw_folder, f"Task{task_id}_{task_name}")
        imagesTr_folder = os.path.join(task_folder, "imagesTr")
        imagesTs_folder = os.path.join(task_folder, "imagesTs")
        labelsTr_folder = os.path.join(task_folder, "labelsTr")
        
        os.makedirs(task_folder, exist_ok=True)
        os.makedirs(imagesTr_folder, exist_ok=True)
        os.makedirs(imagesTs_folder, exist_ok=True)
        os.makedirs(labelsTr_folder, exist_ok=True)
        
        logger.info(f"已创建nnUNet任务文件夹: {task_folder}")
        logger.info("请按照nnUNet要求手动组织数据，或扩展此函数实现自动组织")
        logger.info("nnUNet数据准备完成")
        return True
    except Exception as e:
        logger.error(f"准备nnUNet数据时发生错误: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DCE-MRI预处理 - 用于MAMA-MIA项目的图像预处理")
    parser.add_argument("--input", type=str, required=True, help="输入数据集文件夹路径")
    parser.add_argument("--output", type=str, required=True, help="输出数据集文件夹路径")
    parser.add_argument("--patient", type=str, help="指定要处理的病人ID（可选，默认处理所有病人）")
    parser.add_argument("--phases", type=str, default="0,1,2,3,4", help="要处理的相位列表，以逗号分隔")
    parser.add_argument("--bias-correction", action="store_true", help="是否执行偏置场校正")
    parser.add_argument("--denoise", action="store_true", help="是否执行非局部均值降噪")
    parser.add_argument("--histogram-eq", action="store_true", help="是否执行直方图均衡化")
    parser.add_argument("--clip-percentiles", type=str, default="1,99", help="强度裁剪的百分位数")
    parser.add_argument("--spacing", type=str, default="1,1,1", help="等向素重采样的目标间距")
    parser.add_argument("--nnunet-raw", type=str, help="nnUNet_raw_data_base文件夹路径")
    parser.add_argument("--task-id", type=int, default=101, help="nnUNet任务ID")
    parser.add_argument("--task-name", type=str, default="BreastMRI", help="nnUNet任务名称")
    parser.add_argument("--log-file", type=str, help="日志文件路径（可选）")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="日志级别")
    
    """"""""""""
    #命令行使用
    """"""""""""
    ##   调用添加偏置场的预处理：
        # python pre.py --input D:\Project\MAMA_MIA\code\nnUNet_raw\Task101_BreastMRI\imagesTr --output D:\Project\MAMA_MIA\code\nnUNet_preprocessed --bias-correction --denoise
    ## 调用基础的预处理：
        # python pre.py --input D:\Project\MAMA_MIA\code\nnUNet_raw\Task101_BreastMRI\imagesTr --output D:\Project\MAMA_MIA\code\nnUNet_preprocessed
    ## 转换为nnunet数据格式
        # python pre.py --input D:\Project\MAMA_MIA\code\nnUNet_preprocessed --output D:\Project\MAMA_MIA\code\nnUNet_preprocessed --nnunet-raw D:\Project\MAMA_MIA\code\nnUNet_preprocessed
   # nnUNetv2_predict -i D:\Project\MAMA_MIA\code\nnUNet_raw\Task101_BreastMRI -o D:\Project\MAMA_MIA\code\nnUNet_results -d 101 -c 3d_fullres
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level)
    global logger
    logger = setup_logger(args.log_file, log_level)
    
    phases = [int(p) for p in args.phases.split(",")]
    clip_percentiles = [float(p) for p in args.clip_percentiles.split(",")]
    spacing = [float(s) for s in args.spacing.split(",")]
    
    logger.info(f"启动预处理任务，使用以下参数:")
    logger.info(f"  输入文件夹: {args.input}")
    logger.info(f"  输出文件夹: {args.output}")
    logger.info(f"  要处理的相位: {phases}")
    logger.info(f"  执行偏置场校正: {args.bias_correction}")
    logger.info(f"  执行降噪: {args.denoise}")
    logger.info(f"  执行直方图均衡化: {args.histogram_eq}")
    logger.info(f"  强度裁剪百分位数: {clip_percentiles}")
    logger.info(f"  目标等向素间距: {spacing}")
    
    start_time = time.time()
    
    if args.patient:
        logger.info(f"处理病人 {args.patient}")
        success = preprocess_dce_mri(
            args.input, args.output, args.patient, phases=phases,
            do_bias_correction=args.bias_correction, do_denoise=args.denoise,
            do_histogram_eq=args.histogram_eq, clip_percentiles=clip_percentiles,
            isotropic_spacing=spacing
        )
        if success:
            logger.info(f"成功处理病人 {args.patient}")
        else:
            logger.error(f"处理病人 {args.patient} 失败")
    else:
        logger.info(f"处理整个数据集 {args.input} -> {args.output}")
        success_count = preprocess_dce_mri_dataset(
            args.input, args.output, phases=phases,
            do_bias_correction=args.bias_correction, do_denoise=args.denoise,
            do_histogram_eq=args.histogram_eq, clip_percentiles=clip_percentiles,
            isotropic_spacing=spacing
        )
        logger.info(f"成功处理 {success_count} 个病人")
    
    if args.nnunet_raw:
        logger.info(f"准备nnUNet数据，路径: {args.nnunet_raw}")
        prepare_for_nnunet(args.output, args.nnunet_raw, args.task_id, args.task_name)
    
    end_time = time.time()
    logger.info(f"预处理任务完成，耗时 {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger("MAMA-MIA-Preprocessing")
        logger.critical(f"发生未捕获异常: {str(e)}", exc_info=True)
        sys.exit(1)
