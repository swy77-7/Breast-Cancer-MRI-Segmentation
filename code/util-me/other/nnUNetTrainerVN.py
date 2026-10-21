import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule


class nnUNetTrainerCBAMVNet3D(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Align with nnUNetTrainer defaults
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 500
        self.enable_deep_supervision = True
        self.save_every = 50
        self.disable_checkpointing = False
        self.print_to_log_file("Using CBAMVNet3D architecture for training.")

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build CBAMVNet3D architecture instead of the default PlainConvUNet.
        """
        from nnunetv2.training.nnUNetTrainer.variants.cbam_vnet3d import CBAMVNet3D

        # Define CBAMVNet3D-specific architecture parameters
        arch_init_kwargs = {
            'input_channels': num_input_channels,
            'n_stages': 4,
            'features_per_stage': (16, 32, 64, 128),
            'conv_op': torch.nn.Conv3d,
            'kernel_sizes': 3,
            'strides': (1, 2, 2, 2),
            'n_conv_per_stage': (2, 2, 2, 2),
            'num_classes': num_output_channels,
            'n_conv_per_stage_decoder': (2, 2, 2),
            'conv_bias': True,
            'norm_op': torch.nn.BatchNorm3d,
            'norm_op_kwargs': {'eps': 1e-5, 'momentum': 0.1},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': torch.nn.PReLU,
            'nonlin_kwargs': {},
            'deep_supervision': enable_deep_supervision
        }

        # Instantiate CBAMVNet3D directly
        network = CBAMVNet3D(**arch_init_kwargs)
        network.apply(network.initialize)
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Enable/disable deep supervision in CBAMVNet3D's decoder.
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.decoder.deep_supervision = enabled
        self.print_to_log_file(f"Deep supervision set to: {enabled}")

    def plot_network_architecture(self):
        """
        Override to handle CBAMVNet3D visualization, but keep original behavior.
        """
        if self._do_i_compile():
            self.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
            return

        if self.local_rank == 0:
            try:
                import hiddenlayer as hl
                input_shape = (1, self.num_input_channels, *self.configuration_manager.patch_size)
                g = hl.build_graph(self.network,
                                   torch.rand(input_shape, device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)
            finally:
                empty_cache(self.device)



'''
 ### 代码说明
 1. ** 继承与初始化 **：
- 继承
`nnUNetTrainer`，保留所有原始功能。
- 在
`__init__`
中重置超参数，确保与原始
`nnUNetTrainer.py`
一致（如
`initial_lr = 1e-3
`、`num_epochs = 500
`）。
- 添加日志信息，提示使用
`CBAMVNet3D`。

2. ** build_network_architecture **：
- 重写为静态方法，直接返回
`CBAMVNet3D`
实例。
- 参数设置与
`cbam_vnet3d.py`
一致，基于
`UNet7.py`
的结构：
- `n_stages = 4
`：4
个编码器和
3
个解码器阶段。
- `features_per_stage = (16, 32, 64, 128)
`：逐层增加通道数。
- `strides = (1, 2, 2, 2)
`：下采样策略。
- `n_conv_per_stage = (2, 2, 2, 2)
`：每阶段
2
次卷积。
- `n_conv_per_stage_decoder = (2, 2, 2)
`：解码器每阶段
2
次卷积。
- `deep_supervision = True
`：支持深度监督。
- 使用
`CBAMVNet3D.initialize`
进行
He
初始化。
- 忽略
`architecture_class_name`
和
`arch_init_kwargs_req_import`，直接硬编码
`CBAMVNet3D`
参数（与
nnU - Net
的动态架构加载略有不同，但更适合你的自定义模型）。

3. ** set_deep_supervision_enabled **：
- 重写以适配
`CBAMVNet3D`
的深度监督开关，修改
`decoder.deep_supervision`
属性。
- 兼容
DDP
和
`torch.compile`（`OptimizedModule`）。

4. ** plot_network_architecture **：
- 保留原始逻辑，使用
`hiddenlayer`
绘制网络架构图。
- 确保在
DDP
和编译模式下的兼容性。

5. ** 训练策略一致性 **：
- ** 优化器 **：使用
SGD（`momentum = 0.99
`, `nesterov = True
`）和
`PolyLRScheduler`，与原始一致。
- ** 损失函数 **：复用
`DC_and_CE_loss`（或
`DC_and_BCE_loss`），支持深度监督（`DeepSupervisionWrapper`）。
- ** 数据增强 **：使用原始的
`get_training_transforms`
和
`get_validation_transforms`，包括空间变换、噪声、亮度对比度调整等。
- ** 训练循环 **：完全继承
`train_step`、`validation_step`、`on_epoch_start`
等，确保训练和验证逻辑一致。
- ** 检查点保存 **：每
50
轮保存检查点，保存最佳
EMA
Dice
的模型，与原始一致。
- ** 日志 **：使用
`nnUNetLogger`，记录损失、Dice
等指标。

6. ** 适配乳腺癌
MRI **：
- 参数配置（如
`features_per_stage`）与
`UNet7.py`
一致，适合
3
D
乳腺癌
MRI
分割。
- 支持深度监督，增强复杂边界（如肿瘤）的分割性能。

---

### 使用方法
1. ** 保存代码 **：
- 将
`cbam_vnet3d.py`（前文提供的
artifact_id = "4c151d56-fd32-415c-a50f-d7f27c6615b2"）保存到
`nnunetv2 / training / nnUNetTrainer / variants / cbam_vnet3d.py`。
- 将上述
`nnUNetTrainerCBAMVNet3D.py`
保存到同一目录。

2. ** 数据准备 **：
- 确保
3
D
乳腺癌
MRI
数据集符合
nnU - Net
v2
格式（如
`Dataset300_BreastMRI`）。
- 示例
`dataset.json`（参考前文
artifact_id = "ea010e82-d18e-45c4-8396-320c1b404906"）。

3. ** 环境配置 **：
```bash
export
nnUNet_raw = "/path/to/nnUNet_raw"
export
nnUNet_preprocessed = "/path/to/nnUNet_preprocessed"
export
nnUNet_results = "/path/to/nnUNet_results"
```

4. ** 预处理 **：
```bash
nnUNetv2_plan_and_preprocess - d
300 - -verify_dataset_integrity
```
- 可选：若需自定义
patch
size（如
`(32, 448, 448)`），使用前文的
`ExperimentPlannerCBAMVNet3D`（artifact_id = "146c261a-7fe0-474d-9719-2052a8ef1df2"）：
```bash
nnUNetv2_plan_and_preprocess - d
300 - -verify_dataset_integrity - pl
ExperimentPlannerCBAMVNet3D
```

5. ** 训练 **：
```bash
nnUNetv2_train
300
3
d_fullres
all - tr
nnUNetTrainerCBAMVNet3D
```

6. ** 推理 **：
```bash
nnUNetv2_predict - i / path / to / nnUNet_raw / Dataset300_BreastMRI / imagesTs - o / path / to / output - d
300 - c
3
d_fullres - tr
nnUNetTrainerCBAMVNet3D - f
all
```

7. ** 验证 **：
```bash
nnUNetv2_evaluate_folder - ref / path / to / labelsTs - pred / path / to / output - l
0, 1
```

---

### 验证训练策略一致性
以下是对比，确认
`nnUNetTrainerCBAMVNet3D`
与原始
`nnUNetTrainer`
的训练策略一致：
- ** 优化器 **：SGD，`initial_lr = 1e-3
`, `weight_decay = 3e-5
`, `momentum = 0.99
`, `nesterov = True
`。
- ** 学习率调度 **：`PolyLRScheduler`，随
epoch
递减。
- ** 损失函数 **：
- 分类任务：`DC_and_CE_loss`（Dice + Cross - Entropy）。
- 区域任务：`DC_and_BCE_loss`（Dice + Binary
Cross - Entropy）。
- 深度监督：权重
`[1, 0.5, 0.25, ...]`，最低分辨率权重为
0（DDP
下为
1e-6）。
- ** 数据增强 **：
- 空间变换：旋转（±30° for 3D）、缩放（0.7-1.4）、镜像。
- 强度变换：高斯噪声、模糊、亮度、对比度、伽马变换。
- 级联增强：前景标签的二值操作、随机连通分量移除。
- ** 训练循环 ** ：
- 每 epoch 250 次训练迭代，50 次验证迭代。
- 梯度裁剪（norm=12），支持 `GradScaler`（FP16 训练）。
- ** 检查点 ** ：
- 每 50 轮保存 `checkpoint_latest.pth`。
- 保存最佳 EMA Dice 的 `checkpoint_best.pth`。
- 训练结束保存 `checkpoint_final.pth`。
- ** 验证 ** ：
- 使用 `nnUNetPredictor` 进行滑动窗口推理。
- 计算 Dice、TP / FP / FN，记录到 `nnUNetLogger`。

唯一差异是 `build_network_architecture` 返回 `CBAMVNet3D`，这正是你的需求。

'''