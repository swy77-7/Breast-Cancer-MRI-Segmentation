import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule
import numpy as np
from os.path import join
import time
from typing import Union, List, Tuple
import torch.distributed as dist
from dynamic_network_architectures.architectures.cbamvnet import CBAMVNet3D
from nnunetv2.training.loss.FRLoss import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

class nnUNetTrainerVNet(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
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
        self.best_val_dice = -float('inf')
        self.print_to_log_file("Using CBAMVNet3D architecture with FRLoss for training.")

    def initialize(self):
        super().initialize()
        # Configure the loss function with FRLoss included
        soft_dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
        ce_kwargs = {'ignore_index': None}
        self.loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss
        )
        self.print_to_log_file("Initialized loss function: DC_and_CE_loss with FRLoss")

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Modified to handle (seg_outputs, feature) from CBAMVNet3D and deep supervision.
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        data = data.to(self.device)
        target = [t.to(self.device) for t in target] if isinstance(target, list) else target.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.network(data)
        seg_outputs, feature = outputs  # 解包网络输出

        # 处理深度监督
        if self.enable_deep_supervision and isinstance(seg_outputs, (list, tuple)):
            # 对每个深度监督输出计算损失并加权平均
            loss = 0
            weights = [1, 0.5, 0.25, 0.125, 0.0625][:len(seg_outputs)]  # 权重递减
            for i, output in enumerate(seg_outputs):
                loss += weights[i] * self.loss(feature=feature, net_output=output, target=target[0] if isinstance(target, list) else target)
            loss /= sum(weights)
        else:
            # 单输出情况
            loss = self.loss(feature=feature, net_output=seg_outputs, target=target[0] if isinstance(target, list) else target)

        if do_backprop:
            loss.backward()
            self.optimizer.step()

        if run_online_evaluation:
            # 对于在线评估，使用最高分辨率的输出
            output = seg_outputs[0] if isinstance(seg_outputs, (list, tuple)) else seg_outputs
            self.run_online_evaluation(output, target[0] if isinstance(target, list) else target)

        return loss.item()

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                  arch_init_kwargs: dict,
                                  arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                  num_input_channels: int,
                                  num_output_channels: int,
                                  enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        Build the CBAMVNet3D architecture using parameters from arch_init_kwargs.
        """
        print(f"Building CBAMVNet3D with architecture_class_name={architecture_class_name}")
        # Update arch_init_kwargs with input/output channels and deep supervision
        arch_init_kwargs = arch_init_kwargs.copy()
        arch_init_kwargs['input_channels'] = num_input_channels
        arch_init_kwargs['num_classes'] = num_output_channels
        arch_init_kwargs['deep_supervision'] = enable_deep_supervision

        # Convert string-based module references to actual classes if needed
        for key in arch_init_kwargs_req_import:
            if key in arch_init_kwargs and isinstance(arch_init_kwargs[key], str):
                module_name, class_name = arch_init_kwargs[key].rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                arch_init_kwargs[key] = getattr(module, class_name)

        print(f"Building CBAMVNet3D with n_stages={arch_init_kwargs['n_stages']}, "
              f"features_per_stage={arch_init_kwargs['features_per_stage']}")
        network = CBAMVNet3D(**arch_init_kwargs)
        network.apply(network.initialize)
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Enable/disable deep supervision in the network's decoder.
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
        Plot network architecture visualization, but keep original behavior.
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

    def on_epoch_end(self):
        """
        Handle epoch-end tasks, saving checkpoint_best.pth for best validation Dice, without early stopping.
        """
        self.logger.log('epoch_end_timestamps', time.time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        current_val_dice = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
        self.print_to_log_file(f"Current validation mean foreground Dice: {np.round(current_val_dice, decimals=4)}")

        if current_val_dice > self.best_val_dice:
            self.best_val_dice = current_val_dice
            self.print_to_log_file(f"New best validation Dice: {np.round(self.best_val_dice, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if (self.current_epoch + 1) % self.save_every == 0 and self.current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best_ema.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        """
        Save checkpoint including best validation Dice state.
        """
        if self.local_rank == 0 and not self.disable_checkpointing:
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod

            checkpoint = {
                'network_weights': mod.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                'logging': self.logger.get_checkpoint(),
                '_best_ema': self._best_ema,
                'current_epoch': self.current_epoch + 1,
                'init_args': self.my_init_kwargs,
                'trainer_name': self.__class__.__name__,
                'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                'best_val_dice': self.best_val_dice
            }
            torch.save(checkpoint, filename)
            self.print_to_log_file(f"Saved checkpoint: {filename}")
        elif self.local_rank == 0:
            self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        """
        Load checkpoint including best validation Dice state.
        """
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)

        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes',
                                                              self.inference_allowed_mirroring_axes)
        self.best_val_dice = checkpoint.get('best_val_dice', -float('inf'))

        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None and checkpoint['grad_scaler_state'] is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        self.print_to_log_file(
            f"Loaded checkpoint from {filename_or_checkpoint if isinstance(filename_or_checkpoint, str) else 'checkpoint'}")

    def on_train_end(self):
        """
        Ensure proper cleanup for distributed training.
        """
        super().on_train_end()
        if self.is_ddp and dist.is_initialized():
            dist.destroy_process_group()