from .train_config import TrainConfig
from .dataset import to_dataset
from .lora import to_lora
from .trainer import to_trainer

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


def train(text_or_dataset, cfg: TrainConfig): 
    # preload the model and tokenizer    
    model_ = AutoModelForCausalLM.from_pretrained(cfg.model_preset)
    lora_model = to_lora(model_, **cfg.lora_params)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_preset)

    # preprocess data => tokenized dataset
    if type(text_or_dataset) == str: 
        dataset = to_dataset(text_or_dataset, 
                             tokenizer = tokenizer, 
                             chunk_size=cfg.chunk_size,
                             verbose = True)
    else: 
        dataset = text_or_dataset

    trainer = to_trainer(model = lora_model, 
                         tokenizer = tokenizer, 
                         train_dataset = dataset, 
                         **cfg.training_params)
    
    trainer.train()
    return lora_model


        


