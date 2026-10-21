import os
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage import restoration


def read_mri_phase_from_patient_id(images_folder, patient_id, phase=0):
    '''Read the MRI phase from the selected patient ID using SimpleITK:
    images_folder: str - the path to the folder containing the MRI images
    patient_id: str - the patient id
    phase: int - the phase of the MRI image (0: pre-contrast, 1: first post-contrast, 2: second post-contrast, ...)
    '''
    phase_sitk = sitk.ReadImage(f'{images_folder}/{patient_id}/{patient_id}_000{phase}.nii.gz', sitk.sitkFloat32)
    print(f"Input image spacing: {phase_sitk.GetSpacing()}")
    return phase_sitk


def read_segmentation_from_patient_id(segmentations_folder, patient_id):
    '''Read the segmentation from the selected patient ID using SimpleITK:
    images_folder: str - the path to the folder containing the primary tumor segmentations
    patient_id: str - the patient id
    '''
    mask_sitk = sitk.ReadImage(f'{segmentations_folder}/{patient_id}.nii.gz', sitk.sitkUInt8)
    return mask_sitk


def get_image_orientation_from_direction(img_sitk):
    """
    img_sitk = SimpleITK image
    return string orientation
    """
    filter = sitk.DICOMOrientImageFilter()
    filter.Execute(img_sitk)
    orientation = filter.GetOrientationFromDirectionCosines(img_sitk.GetDirection())
    return orientation


def bias_correction_sitk(image_sitk, otsu_threshold=False, shrink_factor=0):
    """Apply N4 Bias Correction."""
    # 检查并修复spacing
    image_sitk = validate_and_fix_spacing(image_sitk)
    # 检查是否为向量图像
    is_vector = image_sitk.GetNumberOfComponentsPerPixel() > 1
    """
    if shrink_factor:
        # N4BiasFieldCorrectionImageFilter takes too long to run, shrink image
        mask_breast = sitk.OtsuThreshold(image_sitk, 0, 1)
        shrinked_image_sitk = sitk.Shrink(image_sitk, [shrink_factor] * image_sitk.GetDimension())
        shrinked_mask_breast = sitk.Shrink(mask_breast, [shrink_factor] * mask_breast.GetDimension())
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        tmp_image = corrector.Execute(shrinked_image_sitk, shrinked_mask_breast)
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
        corrected_image_sitk = image_sitk / sitk.Exp(log_bias_field)
    else:
        initial_img = image_sitk
        # Cast to float to enable bias correction to be used
        tmp_image = sitk.Cast(image_sitk, sitk.sitkFloat64)
        # Set zeroes to a small number to prevent division by zero
        tmp_image = sitk.GetArrayFromImage(tmp_image)
        tmp_image[tmp_image == 0] = np.finfo(float).eps
        tmp_image = sitk.GetImageFromArray(tmp_image)
        tmp_image.CopyInformation(initial_img)
        if otsu_threshold:
            maskImage = sitk.OtsuThreshold(tmp_image, 0, 1)
        # Apply image bias correction using N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        if otsu_threshold:
            corrected_image_sitk = corrector.Execute(tmp_image, maskImage)
        else:
            corrected_image_sitk = corrector.Execute(tmp_image)
    return corrected_image_sitk
    """
    if shrink_factor:
        # 处理带shrink_factor的情况
        if is_vector:
            # 对向量图像的每个通道分别处理
            components = []
            for i in range(image_sitk.GetNumberOfComponentsPerPixel()):
                component = sitk.VectorIndexSelectionCast(image_sitk, i)
                corrected_component = process_scalar_image(component, otsu_threshold, shrink_factor)
                components.append(corrected_component)
            return sitk.Compose(components)
        else:
            # 标量图像正常处理
            return process_scalar_image(image_sitk, otsu_threshold, shrink_factor)
    else:
        # 处理不带shrink_factor的情况
        if is_vector:
            components = []
            for i in range(image_sitk.GetNumberOfComponentsPerPixel()):
                component = sitk.VectorIndexSelectionCast(image_sitk, i)
                # 对每个通道执行N4校正
                corrected_component = process_scalar_image(component, otsu_threshold, shrink_factor=0)
                components.append(corrected_component)
            return sitk.Compose(components)
        else:
            return process_scalar_image(image_sitk, otsu_threshold, shrink_factor=0)


def validate_and_fix_spacing(image_sitk):
    spacing = image_sitk.GetSpacing()
    new_spacing = list(spacing)
    has_zero = False
    for i, s in enumerate(spacing):
        if s <= 0:
            has_zero = True
            if i > 0 and spacing[i - 1] > 0:
                new_spacing[i] = spacing[i - 1]
                print(f"Warning: Spacing in dimension {i} was {s}, fixed to {new_spacing[i]}")
            elif i < len(spacing) - 1 and spacing[i + 1] > 0:
                new_spacing[i] = spacing[i + 1]
                print(f"Warning: Spacing in dimension {i} was {s}, fixed to {new_spacing[i]}")
            else:
                new_spacing[i] = 1.0
                print(f"Warning: Spacing in dimension {i} was {s}, fixed to default 1.0")
    if has_zero:
        print(f"Warning: Image spacing contained zero values {spacing}, fixed to {new_spacing}")
        image_sitk.SetSpacing(tuple(new_spacing))
    final_spacing = image_sitk.GetSpacing()
    if any(s <= 0 for s in final_spacing):
        raise RuntimeError(f"Fatal error: Spacing still contains zero values after fix: {final_spacing}")
    return image_sitk

def process_scalar_image(image_sitk, otsu_threshold, shrink_factor):
    """Process scalar (single-channel) images with N4 bias correction."""
    # 再次验证spacing，确保安全
    image_sitk = validate_and_fix_spacing(image_sitk)
    print(f"Spacing before N4 correction: {image_sitk.GetSpacing()}")

    initial_img = image_sitk

    if shrink_factor > 0:
        # 使用shrink_factor加速处理
        mask_breast = sitk.OtsuThreshold(image_sitk, 0, 1)
        shrinked_image_sitk = sitk.Shrink(image_sitk, [shrink_factor] * image_sitk.GetDimension())
        shrinked_mask_breast = sitk.Shrink(mask_breast, [shrink_factor] * mask_breast.GetDimension())

        # 修复缩小后的 spacing
        shrinked_spacing = tuple(s * shrink_factor for s in image_sitk.GetSpacing())
        shrinked_spacing = tuple(max(s, 0.001) for s in shrinked_spacing)  # 防止零值
        shrinked_image_sitk.SetSpacing(shrinked_spacing)
        shrinked_mask_breast.SetSpacing(shrinked_spacing)
        print(f"Shrinked spacing: {shrinked_image_sitk.GetSpacing()}")

        # 确保图像是Float64类型
        shrinked_image_sitk = sitk.Cast(shrinked_image_sitk, sitk.sitkFloat64)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        tmp_image = corrector.Execute(shrinked_image_sitk, shrinked_mask_breast)

        # 获取对数偏置场并扩展回原始大小
        log_bias_field = corrector.GetLogBiasFieldAsImage(image_sitk)
        corrected_image_sitk = image_sitk / sitk.Exp(log_bias_field)

        # 确保输出为Float32类型
        # return sitk.Cast(corrected_image_sitk, sitk.sitkFloat32)
    else:
        # 不使用shrink_factor的处理
        # 先转换为Float64进行处理
        try:
            tmp_image = sitk.Cast(image_sitk, sitk.sitkFloat64)
        except RuntimeError as e:
            print(f"Error casting image to Float64: {e}")
            # 如果转换失败，尝试使用Float32
            tmp_image = sitk.Cast(image_sitk, sitk.sitkFloat32)

        # 设置零值为一个小正数，防止除以零
        # 注意：直接操作数组可能破坏向量结构，这里使用SimpleITK方法
        if otsu_threshold:
            maskImage = sitk.OtsuThreshold(tmp_image, 0, 1)

        # 应用N4偏置校正
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        if otsu_threshold:
            corrected_image_sitk = corrector.Execute(tmp_image, maskImage)
        else:
            corrected_image_sitk = corrector.Execute(tmp_image)

    # 验证输出 spacing
    final_spacing = corrected_image_sitk.GetSpacing()
    if any(s <= 0 for s in final_spacing):
        raise RuntimeError(f"Invalid spacing after N4 correction: {final_spacing}")

    # 确保输出为Float32类型
    return sitk.Cast(corrected_image_sitk, sitk.sitkFloat32)

def nlmeans_denoise_sitk(image_sitk, patch_size=5, patch_distance=6, h=0.8):
    """
    Denoises a DCE-MRI image using Non-Local Means (NLMeans) filtering.

    Parameters:
        image_sitk (SimpleITK.Image): Input DCE-MRI image.
        patch_size (int): Size of the patches used for denoising.
        patch_distance (int): Maximal distance in pixels where to search patches used for denoising.
        h (float): Cut-off distance (higher h means more smoothing).

    Returns:
        SimpleITK.Image: Denoised DCE-MRI image.
    """
    # Convert SimpleITK image to NumPy array
    image_np = sitk.GetArrayFromImage(image_sitk)

    # Apply Non-Local Means denoising
    denoised_np = restoration.denoise_nl_means(
        image_np,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        fast_mode=True
    )

    # Convert denoised NumPy array back to SimpleITK image
    denoised_image_sitk = sitk.GetImageFromArray(denoised_np)
    denoised_image_sitk.CopyInformation(image_sitk)  # Preserve original metadata

    return denoised_image_sitk


def clip_image_sitk(image_sitk, percentiles=[1, 99]):
    """Clip intensity range of an image.

    Parameters
    image: ITK Image
        Image to normalize
    lowerbound: float, default -1000.0
        lower bound of clipping range
    upperbound: float, default 3000.0
        lower bound of clipping range

    Returns
    -------
    image : ITK Image
        Output image.

    """
    image_array = sitk.GetArrayFromImage(image_sitk)
    image_array = image_array.ravel()
    # Drop all zeroes from array
    image_array = image_array[image_array != 0]
    lowerbound = np.percentile(image_array, percentiles[0])
    upperbound = np.percentile(image_array, percentiles[1])
    # Create clamping filter for clipping and set variables
    filter = sitk.ClampImageFilter()
    filter.SetLowerBound(float(lowerbound))
    filter.SetUpperBound(float(upperbound))

    # Execute
    clipped_image_sitk = filter.Execute(image_sitk)

    return clipped_image_sitk


def zscore_normalization_sitk(image_sitk, mean, std):
    # Z-score normalization
    array = sitk.GetArrayFromImage(image_sitk)
    normalized_array = (array - mean) / std
    zscored_sitk = sitk.GetImageFromArray(normalized_array)
    zscored_sitk.CopyInformation(image_sitk)
    return zscored_sitk


def histogram_equalization_sitk(image_sitk, alpha=0.3, beta=0.3, radius=5):
    """Perform histogram equalization on an image.

    See for documentation https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1AdaptiveHistogramEqualizationImageFilter.html.

     Parameters
    ----------
    image_sitk: ITK Image
        Image to normalize

    alpha: float, default 0.3
        controls how much the filter acts like the classical histogram equalization
        method (alpha=0) to how much the filter acts like an unsharp mask (alpha=1).

    beta: float, default 0.3
        controls how much the filter acts like an unsharp mask (beta=0) to much
        the filter acts like pass through (beta=1, with alpha=1).

    radius: integer, default None
       Controls the size of the region over which local statistics are calculated.
       The size of the window is controlled by SetRadius the default Radius is 5 in all directions.

    Returns
    -------
    image : ITK Image
        Output image.

    """
    hqfilter = sitk.AdaptiveHistogramEqualizationImageFilter()
    hqfilter.SetAlpha(alpha)
    hqfilter.SetBeta(beta)
    hqfilter.SetRadius(radius)
    heq_image = hqfilter.Execute(image_sitk)
    return heq_image


def resample_sitk(image_sitk, new_spacing=None, new_size=None,
                  interpolator=sitk.sitkBSpline, tol=0.00001):
    # Get original settings
    original_size = image_sitk.GetSize()
    original_spacing = image_sitk.GetSpacing()

    # ITK can only do 3D images
    if len(original_size) == 2:
        original_size = original_size + (1,)
    if len(original_spacing) == 2:
        original_spacing = original_spacing + (1.0,)

    # 验证输入 spacing
    image_sitk = validate_and_fix_spacing(image_sitk, image_path="resample_input")

    if new_spacing is None and new_size is None:
        raise ValueError("Either new_spacing or new_size must be provided")

    if new_spacing is not None:
        # 确保 new_spacing 不包含零值
        new_spacing = list(new_spacing)
        for i, s in enumerate(new_spacing):
            if s <= 0:
                new_spacing[i] = original_spacing[i] if original_spacing[i] > 0 else 1.0
                print(f"警告: new_spacing 维度 {i} 为 {s}，已修复为 {new_spacing[i]}")

    if new_size is None:
        # Compute output size
        new_size = [round(original_size[i] * (original_spacing[i] + tol) / new_spacing[i])
                    for i in range(len(original_size))]
        new_size = [max(1, s) for s in new_size]  # 确保尺寸不为零

    if new_spacing is None:
        # Compute output spacing
        new_spacing = [original_size[i] * original_spacing[i] / new_size[i]
                       for i in range(len(original_size))]
        new_spacing = [max(s, 0.001) for s in new_spacing]  # 防止零值

    # 再次验证 spacing
    if any(s <= 0 for s in new_spacing):
        raise ValueError(f"Invalid new_spacing: {new_spacing}")

    # Set and execute the filter
    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(new_spacing)
    ResampleFilter.SetSize(np.array(new_size, dtype='int').tolist())
    ResampleFilter.SetOutputDirection(image_sitk.GetDirection())
    ResampleFilter.SetOutputOrigin(image_sitk.GetOrigin())
    ResampleFilter.SetOutputPixelType(image_sitk.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    try:
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)
    except RuntimeError:
        # Assume the error is due to the direction determinant being 0
        # Solution: simply set a correct direction
        # print('Bad output direction in resampling, resetting direction.')
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ResampleFilter.SetOutputDirection(direction)
        image_sitk.SetDirection(direction)
        resampled_image_sitk = ResampleFilter.Execute(image_sitk)

    # 验证输出 spacing
    final_spacing = resampled_image_sitk.GetSpacing()
    if any(s <= 0 for s in final_spacing):
        raise RuntimeError(f"Resampled image has invalid spacing: {final_spacing}")

    return resampled_image_sitk