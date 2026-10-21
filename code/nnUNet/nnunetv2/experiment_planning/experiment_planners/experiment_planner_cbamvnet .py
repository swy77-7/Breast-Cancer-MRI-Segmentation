from typing import Union, List, Tuple
import os
import numpy as np
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, maybe_mkdir_p
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

class ExperimentPlannerCBAMVNet3D(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int], gpu_memory_target_in_gb: float = 24.0,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlansCBAMVNet',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.plans_name = plans_name
        self.UNet_class = 'nnunetv2.network_architecture.architectures.cbamvnet.CBAMVNet3D'
        self.UNet_base_num_features = 16
        self.UNet_max_features_3d = 128
        self.UNet_blocks_per_stage_encoder = (2, 2, 2, 2)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2)
        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_3d = 550000000
        self.UNet_min_batch_size = 2
        self.UNet_featuremap_min_edge_length = 4

    def static_estimate_VRAM_usage(self, patch_size: Tuple[int], input_channels: int, output_channels: int,
                                  arch_class_name: str, arch_kwargs: dict, arch_kwargs_req_import: Tuple[str, ...]):
        import torch
        a = torch.get_num_threads()
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        torch.set_num_threads(get_allowed_n_proc_DA())
        try:
            # Use arch_kwargs directly, as output_channels will be passed as num_classes
            net = get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import,
                                         input_channels, output_channels, allow_init=True)
            ret = net.compute_conv_feature_map_size(patch_size)
        except Exception as e:
            print(f"VRAM estimation failed: {e}")
            raise e
        finally:
            torch.set_num_threads(a)
        return ret

    def get_plans_for_configuration(self, spacing: Union[np.ndarray, List[float]],
                                    median_shape: Union[np.ndarray, List[int]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        assert all(isinstance(i, (int, float)) and i > 0 for i in spacing), f"Invalid spacing: {spacing}"
        num_input_channels = len(self.dataset_json.get('channel_names', self.dataset_json.get('modality', {})))
        num_output_classes = len(self.dataset_json['labels']) - 1  # Exclude background
        unet_conv_op = convert_dim_to_conv_op(len(spacing))
        norm_op = get_matching_instancenorm(unet_conv_op)

        # Initialize patch size based on spacing aspect ratio
        tmp = 1 / np.array(spacing)
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        else:
            raise ValueError("3D configuration expected for CBAMVNet3D")
        initial_patch_size = [min(p, m) for p, m in zip(initial_patch_size, median_shape)]

        # Compute network topology
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, shape_must_be_divisible_by = \
            get_pool_and_conv_props(spacing, initial_patch_size, self.UNet_featuremap_min_edge_length, 999999)
        num_stages = len(pool_op_kernel_sizes)

        # Define CBAMVNet3D architecture parameters, excluding num_classes
        features_per_stage = [min(self.UNet_max_features_3d, self.UNet_base_num_features * 2 ** i) for i in range(num_stages)]
        arch_kwargs = {
            'n_stages': num_stages,
            'features_per_stage': features_per_stage,
            'conv_op': 'torch.nn.Conv3d',
            'kernel_sizes': conv_kernel_sizes if len(conv_kernel_sizes) == num_stages else 3,
            'strides': pool_op_kernel_sizes,
            'n_conv_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
            'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            'conv_bias': True,
            'norm_op': 'torch.nn.BatchNorm3d',
            'norm_op_kwargs': {'eps': 1e-5, 'momentum': 0.1},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.PReLU',
            'nonlin_kwargs': {},
            'deep_supervision': True
        }

        architecture_kwargs = {
            'network_class_name': self.UNet_class,
            'arch_kwargs': arch_kwargs,
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin')
        }

        # Estimate VRAM usage
        estimate = self.static_estimate_VRAM_usage(patch_size, num_input_channels, num_output_classes,
                                                  architecture_kwargs['network_class_name'],
                                                  architecture_kwargs['arch_kwargs'],
                                                  architecture_kwargs['_kw_requires_import'])
        _cache[str(patch_size) + '_' + str(pool_op_kernel_sizes)] = estimate

        # Adjust patch size based on VRAM target
        reference = self.UNet_reference_val_3d * (self.UNet_vram_target_GB / 8)
        ref_bs = 2
        while (estimate / ref_bs * 2) > reference:
            axis_to_be_reduced = np.argsort([p / m for p, m in zip(patch_size, median_shape)])[-1]
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, tmp_patch_size, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp, self.UNet_featuremap_min_edge_length, 999999)
            patch_size[axis_to_be_reduced] = tmp_patch_size[axis_to_be_reduced]

            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, patch_size, self.UNet_featuremap_min_edge_length, 999999)
            num_stages = len(pool_op_kernel_sizes)
            arch_kwargs['n_stages'] = num_stages
            arch_kwargs['features_per_stage'] = [min(self.UNet_max_features_3d, self.UNet_base_num_features * 2 ** i)
                                                for i in range(num_stages)]
            arch_kwargs['strides'] = pool_op_kernel_sizes
            arch_kwargs['kernel_sizes'] = conv_kernel_sizes if len(conv_kernel_sizes) == num_stages else 3
            arch_kwargs['n_conv_per_stage'] = self.UNet_blocks_per_stage_encoder[:num_stages]
            arch_kwargs['n_conv_per_stage_decoder'] = self.UNet_blocks_per_stage_decoder[:num_stages - 1]
            architecture_kwargs['arch_kwargs'] = arch_kwargs

            estimate = self.static_estimate_VRAM_usage(patch_size, num_input_channels, num_output_classes,
                                                      architecture_kwargs['network_class_name'],
                                                      architecture_kwargs['arch_kwargs'],
                                                      architecture_kwargs['_kw_requires_import'])
            _cache[str(patch_size) + '_' + str(pool_op_kernel_sizes)] = estimate

        # Compute batch size
        batch_size = max(round((reference / estimate) * ref_bs), self.UNet_min_batch_size)
        bs_max_5_percent = round(approximate_n_voxels_dataset * 0.05 / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_max_5_percent), self.UNet_min_batch_size)

        # Resampling and normalization
        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()
        normalization_schemes, use_nonzero_mask_for_norm = self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        return {
            'inherits_from': '3d_fullres',
            'network_trainer_class_name': 'nnUNetTrainerVNet',
            'data_identifier': data_identifier,
            'preprocessor_name': self.preprocessor_name,
            'batch_size': batch_size,
            'patch_size': patch_size.tolist(),
            'median_image_size_in_voxels': [int(round(i)) for i in median_shape],
            'spacing': [float(i) for i in spacing],
            'normalization_schemes': normalization_schemes,
            'use_mask_for_norm': use_nonzero_mask_for_norm,
            'resampling_fn_data': resampling_data.__name__,
            'resampling_fn_seg': resampling_seg.__name__,
            'resampling_fn_data_kwargs': resampling_data_kwargs,
            'resampling_fn_seg_kwargs': resampling_seg_kwargs,
            'resampling_fn_probabilities': resampling_softmax.__name__,
            'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
            'network_architecture': architecture_kwargs
        }

    def plan_experiment(self):
        _cache = {}
        transpose_forward, transpose_backward = self.determine_transpose()
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed) * self.dataset_json['numTraining'])

        plan_3d_fullres = self.get_plans_for_configuration(
            fullres_spacing_transposed,
            new_median_shape_transposed,
            self.generate_data_identifier('3d_fullres'),
            approximate_n_voxels_dataset,
            _cache
        )
        plan_3d_fullres['batch_dice'] = False

        # Add custom 3d_fullres_cbamvnet configuration
        plan_3d_cbamvnet = self.get_plans_for_configuration(
            fullres_spacing_transposed,
            new_median_shape_transposed,
            self.generate_data_identifier('3d_fullres_cbamvnet'),
            approximate_n_voxels_dataset,
            _cache
        )
        plan_3d_cbamvnet['batch_dice'] = False

        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        import shutil
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(self.nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_name,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {
                '2d': self.get_plans_for_configuration(
                    fullres_spacing_transposed[1:],
                    new_median_shape_transposed[1:],
                    self.generate_data_identifier('2d'),
                    approximate_n_voxels_dataset,
                    _cache
                ),
                '3d_fullres': plan_3d_fullres,
                '3d_fullres_cbamvnet': plan_3d_cbamvnet
            },
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint['foreground_intensity_properties_per_channel']
        }

        self.plans = plans
        self.save_plans(plans)
        return plans