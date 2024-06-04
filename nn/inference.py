from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

class ImportModel:
    def __init__(self,
                model_path,
                generation_config: GenerationConfig = GenerationConfig(),
                trainable = False) -> PeftModel:
        """

        Args:
        model_path: path to model checkpoint 
        generation_config: GenerationConfig transformers class 
        trainable: default = False. If true is able to train model again

        Returns:
          Loaded model 
        """

        self.model_path = model_path
        self.gen_cfg = generation_config
        # Configure device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the PEFT configuration
        self.peft_config = PeftConfig.from_pretrained(model_path)

        # Load the base model using the base model path from the PEFT configuration
        self.base_model = AutoModelForCausalLM.from_pretrained(self.peft_config.base_model_name_or_path)

        # Load the tokenizer using the base model path from the PEFT configuration
        self.tokenizer = AutoTokenizer.from_pretrained(self.peft_config.base_model_name_or_path)

        # Load the final PEFT model with the LoRA weights
        self.model = PeftModel.from_pretrained(self.base_model, model_path, trainable = trainable).to(self.device)


    def __call__(self, inputs: str):
        tokens = self.tokenizer(inputs, return_tensors = 'pt').to(self.device)

        with torch.no_grad(): 
            model_outputs = self.model.generate(**tokens, generation_config = self.gen_cfg)
            decoded = self.tokenizer.batch_decode(model_outputs)
            return decoded
    
def interface(model_path, **cfg_params): 
    """

    Args:
      model_path: path to peft model checkpoint  
    """
    config = GenerationConfig(**cfg_params)
    model = ImportModel(model_path=model_path, generation_config = config)

    def f(): 
      text = str(input("[USER]: "))
      responce = model(text)
      print(f"[MODEL]: {responce}")
      f()
    f()