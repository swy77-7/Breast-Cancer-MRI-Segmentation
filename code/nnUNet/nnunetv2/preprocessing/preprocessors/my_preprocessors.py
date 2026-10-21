import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

'''
class MyPreprocessor:
    def __init__(self, plans_manager: PlansManager, configuration_manager: ConfigurationManager, dataset_json: dict, verbose: bool = True):
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.verbose = verbose

    def run(self, data: np.ndarray, seg: np.ndarray, properties: dict):
        """
        Core preprocessing pipeline: transpose -> crop -> normalize -> resample -> sample foreground.
        """
        # ensure float32
        data = data.astype(np.float32)
        has_seg = seg is not None
        if has_seg:
            seg = seg.copy()

        # transpose
        order = [0] + [i+1 for i in self.plans_manager.transpose_forward]
        data = data.transpose(order)
        if has_seg:
            seg = seg.transpose(order)
        original_spacing = [properties['spacing'][i] for i in self.plans_manager.transpose_forward]

        # crop
        properties['shape_before_cropping'] = data.shape[1:]
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # normalize
        data = self._normalize(data, seg)

        # resample
        target_spacing = self.configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        old_shape = data.shape[1:]
        data = self.configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = self.configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f"Resample: {old_shape}->{new_shape}, {original_spacing}->{target_spacing}")

        # foreground sampling
        if has_seg:
            props = self.plans_manager.get_label_manager(self.dataset_json)
            regions = props.foreground_regions if props.has_regions else props.foreground_labels
            if props.has_ignore_label:
                regions.append([-1] + props.all_labels)
            properties['class_locations'] = self._sample_foreground_locations(seg, regions)
            seg = self.modify_seg(seg)

        return data, seg, properties

    def _normalize(self, data: np.ndarray, seg: np.ndarray) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = self.configuration_manager.normalization_schemes[c]
            normalizer_cls = self._find_normalizer(scheme)
            normalizer = normalizer_cls(use_mask_for_norm=self.configuration_manager.use_mask_for_norm[c],
                                        intensityproperties=self.plans_manager.foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def _find_normalizer(self, scheme: str):
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        import nnunetv2
        path = join(nnunetv2.__path__[0], "preprocessing", "normalization")
        cls = recursive_find_python_class(path, scheme, 'nnunetv2.preprocessing.normalization')
        if cls is None:
            raise RuntimeError(f"Could not find normalizer {scheme}")
        return cls

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions, seed=1234, verbose=False):
        # Implementation of Otsu-based sampling...
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        return DefaultPreprocessor._sample_foreground_locations(seg, classes_or_regions, seed, verbose)

    def modify_seg(self, seg: np.ndarray) -> np.ndarray:
        # Hook to modify segmentation after resampling; default: no change
        return seg
'''

class MyPreprocessor(DefaultPreprocessor):  # 继承自 DefaultPreprocessor
    def __init__(self, plans_manager: PlansManager, configuration_manager: ConfigurationManager, dataset_json: dict, verbose: bool = True):
        super().__init__(verbose)  # 调用父类的初始化方法
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.verbose = verbose

    def run(self, data: np.ndarray, seg: np.ndarray, properties: dict, num_processes: int = 1):
        """
        Core preprocessing pipeline: transpose -> crop -> normalize -> resample -> sample foreground.
        """
        # ensure float32
        data = data.astype(np.float32)
        has_seg = seg is not None
        if has_seg:
            seg = seg.copy()

        # transpose
        order = [0] + [i+1 for i in self.plans_manager.transpose_forward]
        data = data.transpose(order)
        if has_seg:
            seg = seg.transpose(order)
        original_spacing = [properties['spacing'][i] for i in self.plans_manager.transpose_forward]

        # crop
        properties['shape_before_cropping'] = data.shape[1:]
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # normalize
        data = self._normalize(data, seg)

        # resample
        target_spacing = self.configuration_manager.spacing
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        old_shape = data.shape[1:]
        # data = self.configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        # seg = self.configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        # 使用 num_processes 参数进行数据重采样
        data = self.configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing,
                                                             num_processes=num_processes)
        if has_seg:
            # 使用 num_processes 参数进行标签重采样
            seg = self.configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing,
                                                               num_processes=num_processes)

        if self.verbose:
            print(f"Resample: {old_shape}->{new_shape}, {original_spacing}->{target_spacing}")

        # foreground sampling
        if has_seg:
            props = self.plans_manager.get_label_manager(self.dataset_json)
            regions = props.foreground_regions if props.has_regions else props.foreground_labels
            if props.has_ignore_label:
                regions.append([-1] + props.all_labels)
            properties['class_locations'] = self._sample_foreground_locations(seg, regions)
            seg = self.modify_seg(seg)

        return data, seg, properties

    def _normalize(self, data: np.ndarray, seg: np.ndarray) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = self.configuration_manager.normalization_schemes[c]
            normalizer_cls = self._find_normalizer(scheme)
            normalizer = normalizer_cls(use_mask_for_norm=self.configuration_manager.use_mask_for_norm[c],
                                        intensityproperties=self.plans_manager.foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def _find_normalizer(self, scheme: str):
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
        import nnunetv2
        path = join(nnunetv2.__path__[0], "preprocessing", "normalization")
        cls = recursive_find_python_class(path, scheme, 'nnunetv2.preprocessing.normalization')
        if cls is None:
            raise RuntimeError(f"Could not find normalizer {scheme}")
        return cls

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions, seed=1234, verbose=False):
        # Implementation of Otsu-based sampling...
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
        return DefaultPreprocessor._sample_foreground_locations(seg, classes_or_regions, seed, verbose)

    def modify_seg(self, seg: np.ndarray) -> np.ndarray:
        # Hook to modify segmentation after resampling; default: no change
        return seg