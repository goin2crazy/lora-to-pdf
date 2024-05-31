import torch 

class TrainConfig():
  chunk_size = 150

  model_preset = "0x7o/nanoFialka-v1"
  model_save_path=f"{ model_preset.split(r'/')[-1] }-lora"

  # lora rank, lora rank alpha, target_modules (depends on model), ...
  lora_params = dict(
      r=8,
      lora_alpha=32,
      lora_dropout=0.25,
      target_modules=['lm_head', 'c_fc', 'c_proj', 'wte', 'wpe']
      )

  # epochs, batch_size, precision, ...
  training_params = dict(
    output_dir=model_save_path,
    num_train_epochs = 30,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=10,
    learning_rate=1e-4,
    weight_decay=1e-3,
    fp16 = True,
    push_to_hub=False
)

  # TOKENS INITIALIZATION
  def __init__(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')