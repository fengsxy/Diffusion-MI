from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning as L
import torch
from typing import Dict, Any


class TrainerFactory:
    
    @staticmethod
    def configure_trainer(trainer_config: Dict[str, Any], 
                          model_name: str,
                          task_name: str, 
                          logger_name: str, 
                          train_sample_num: int, 
                          test_num: int) -> L.Trainer:
        callbacks = []

        # Configure Early Stopping
        if 'early_stopping' in trainer_config and trainer_config['early_stopping'] is not None:
            print(trainer_config['early_stopping']) 
            early_stop_callback = EarlyStopping(**trainer_config['early_stopping'])
            callbacks.append(early_stop_callback)

        # Configure Checkpointing (optional)
        if 'checkpoint' in trainer_config and trainer_config['checkpoint'] is not None:
            checkpoint_config = trainer_config['checkpoint'].copy()
            checkpoint_config['filename'] = checkpoint_config['filename'].format(
                model_name=model_name,
                logger_name=logger_name,
                train_sample_num=train_sample_num
            )
            checkpoint_config['dirpath'] = f"{checkpoint_config['dirpath']}/{task_name}"
            checkpoint_callback = ModelCheckpoint(**checkpoint_config)
            callbacks.append(checkpoint_callback)

        # No external logger: we always pass logger=False to the Trainer.
        logger = False

        # Configure Trainer; default to single-device training
        # (no DDP) unless explicitly overridden via ``trainer``
        # kwargs.
        trainer_kwargs = trainer_config.get('trainer', {}).copy()
        use_gpu = torch.cuda.is_available()
        trainer_kwargs.setdefault('accelerator', 'gpu' if use_gpu else 'cpu')
        trainer_kwargs.setdefault('devices', 1)
        trainer_kwargs.update({
            'callbacks': callbacks,
            'logger': logger,
            # Disable Lightning's default checkpointing callback to avoid
            # filesystem permission and cross-device issues; explicit
            # ModelCheckpoint callbacks above still function normally.
            'enable_checkpointing': False,
        })

        return L.Trainer(**trainer_kwargs)


    @staticmethod
    def detect_warnings(smoothed_mi_history):## TODO Do we want the max mi or min loss? 
        warnings = {}
        if len(smoothed_mi_history) < 2:
            return warnings
        
        final_mi = smoothed_mi_history[-1]
        max_mi = max(smoothed_mi_history)
        
        if max_mi > 1.05 * final_mi:
            warnings['max_training_mi_decreased'] = f"WARNING: Smoothed training MI fell compared to highest value: max={max_mi:.3f} vs final={final_mi:.3f}"
        
        if final_mi > 1.05 * smoothed_mi_history[-2]:
            warnings['training_mi_still_increasing'] = f"WARNING: Smoothed training MI was still increasing when training stopped: final={final_mi:.3f} vs previous={smoothed_mi_history[-2]:.3f}"
        
        return warnings
