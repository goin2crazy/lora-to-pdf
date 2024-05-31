from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def to_lora(model: AutoModelForCausalLM, **lora_params):
    # Define the LoRA configuration
    lora_config = LoraConfig(
        **lora_params,  # Assuming cfg.lora_params contains the necessary parameters
        task_type=TaskType.CAUSAL_LM,  # Specify the task type for causal language modeling
    )

    # Apply the LoRA configuration to the model
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    return lora_model 