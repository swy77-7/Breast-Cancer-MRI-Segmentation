import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule
import numpy as np
from os.path import join
import time
from typing import Union
import torch.distributed as dist

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
        self.print_to_log_file("Using CBAMVNet3D architecture for training.")

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
        # Use self.logger correctly as an nnUNetLogger instance
        self.logger.log('epoch_end_timestamps', time.time(), self.current_epoch)

        # Log metrics
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # Track best validation Dice
        current_val_dice = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
        self.print_to_log_file(f"Current validation mean foreground Dice: {np.round(current_val_dice, decimals=4)}")

        if current_val_dice > self.best_val_dice:
            self.best_val_dice = current_val_dice
            self.print_to_log_file(f"New best validation Dice: {np.round(self.best_val_dice, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        # Periodic checkpointing
        if (self.current_epoch + 1) % self.save_every == 0 and self.current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # Update EMA Dice
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