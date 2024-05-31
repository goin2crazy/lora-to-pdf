from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer 
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset 

def to_trainer(lora_model, 
               tokenizer:AutoTokenizer, 
               train_dataset: Dataset, 
               **training_params): 
    
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        **training_params,

    )

    # Create a Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    return trainer