from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from batchgenerators.utilities.file_and_folder_operations import join, load_json
import numpy as np
import os

class ExperimentPlannerCBAMVNet3D(ExperimentPlanner):
    def __init__(self, dataset_id: int, preprocessor_name: str = 'DefaultPreprocessor',
                 overwrite_target_spacing=None, suppress_transpose: bool = False,
                 plans_name: str = 'nnUNetPlansCBAMVNet', gpu_memory_target_in_gb: float = None):
        gpu_memory_target_in_gb = gpu_memory_target_in_gb if gpu_memory_target_in_gb is not None else 24.0
        super().__init__(dataset_name_or_id=dataset_id,
                         preprocessor_name=preprocessor_name,
                         overwrite_target_spacing=overwrite_target_spacing,
                         suppress_transpose=suppress_transpose,
                         plans_name=plans_name,
                         gpu_memory_target_in_gb=gpu_memory_target_in_gb)
        self.dataset_id = dataset_id
        self.plans_name = plans_name

    def plan_experiment(self):
        """
        Generate plans for CBAMVNet3D with 3d_fullres_cbamvnet configuration.
        """
        plans = super().plan_experiment()

        nnUNet_raw = os.environ.get('nnUNet_raw')
        if nnUNet_raw is None:
            raise ValueError("Environment variable nnUNet_raw is not set. Please set it to the path of your raw data directory.")
        dataset_json_path = join(nnUNet_raw, self.dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_path)
        num_classes = len(dataset_json['labels']) - 1

        configuration_name = '3d_fullres_cbamvnet'
        base_config = plans['configurations'].get('3d_fullres', {})

        patch_size = [128, 128, 128]
        batch_size = 2
        target_spacing = base_config.get('spacing', [1.0, 1.0, 1.0])

        arch_kwargs = {
            'n_stages': 4,
            'features_per_stage': [16, 32, 64, 128],
            'conv_op': 'torch.nn.Conv3d',
            'norm_op': 'torch.nn.BatchNorm3d',
            'nonlin': 'torch.nn.PReLU',
            'n_conv_per_stage': [2, 2, 2, 2],
            'n_conv_per_stage_decoder': [2, 2, 2],
            'num_classes': num_classes,
            'deep_supervision': True,
            'conv_bias': False,
            'dropout_op': None,
            'nonlin_first': False
        }

        new_config = {
            'batch_size': batch_size,
            'patch_size': patch_size,
            'spacing': target_spacing,
            'network_class_name': 'nnunetv2.network_architecture.architectures.cbamvnet.CBAMVNet3D',
            'arch_kwargs': arch_kwargs,
            'preprocessor_class_name': self.preprocessor_name,
            'data_identifier': f'{self.plans_name}_{configuration_name}',
            'normalization_schemes': base_config.get('normalization_schemes', ['CTNormalization']),
            'median_image_size_in_voxels': base_config.get('median_image_size_in_voxels', patch_size),
            'resize_to': patch_size,
            'resampling_fn_data': base_config.get('resampling_fn_data', 'resample_3d'),
            'resampling_fn_data_kwargs': base_config.get('resampling_fn_data_kwargs', {'order': 3}),
            'resampling_fn_seg': base_config.get('resampling_fn_seg', 'resample_3d'),
            'resampling_fn_seg_kwargs': base_config.get('resampling_fn_seg_kwargs', {'order': 1}),
            'resampling_fn_probabilities': base_config.get('resampling_fn_probabilities', 'resample_3d'),
            'resampling_fn_probabilities_kwargs': base_config.get('resampling_fn_probabilities_kwargs', {'order': 1}),
            'intensity_properties': base_config.get('intensity_properties', {}),
        }

        plans['configurations'][configuration_name] = new_config
        plans['plans_name'] = self.plans_name
        plans['dataset_name'] = self.dataset_name

        return plans