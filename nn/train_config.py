import torch

class TrainConfig:
    def __init__(self, 
                 chunk_size=150, 
                 model_preset="0x7o/nanoFialka-v1", 
                 model_save_path: str = '', 
                 lora_params=None, 
                 training_params=None, 
                 # LORA PARAMS
                 rank=8, 
                 l_alpha=24, 
                 l_dropout=0.1, 
                 l_modules=None, 
                 # TRAIN PARAMS
                 num_epoch=10, 
                 batch_size=1, 
                 save_steps=50, 
                 save_total_limit=1, 
                 lr=1e-4, 
                 wd=1e-3):
        self.chunk_size = chunk_size

        self.model_preset = model_preset

        self.model_save_path = f"{self.model_preset.split(r'/')[-1]}-lora" if len(model_save_path) else model_save_path

        if l_modules is None:
            l_modules = ['lm_head', 'c_fc', 'c_proj', 'wte', 'wpe']

        # lora rank, lora rank alpha, target_modules (depends on model), ...
        self.lora_params = lora_params if lora_params is not None else dict(
            r=rank,
            lora_alpha=l_alpha,
            lora_dropout=l_dropout,
            target_modules=l_modules
        )

        # epochs, batch_size, precision, ...
        self.training_params = training_params if training_params is not None else dict(
            output_dir=self.model_save_path,
            num_train_epochs=num_epoch,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            learning_rate=lr,
            weight_decay=wd,
            push_to_hub=False
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            self.training_params['fp16'] = True
