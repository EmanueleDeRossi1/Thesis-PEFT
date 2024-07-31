from peft import get_peft_model, LoraConfig, TaskType
from transformers import BertModel

def LoRA_model(base_model_name = "bert-base-multilingual-cased"):
    
    base_model = BertModel.from_pretrained(base_model_name)
    
    # r: Lora attention dimension (the rank). The number of dimensions one matrix
    # lora_alpha: the alpha paramter for Lora scaling. It controls how influence the low-rank matrices have on the model during fine-tuning
    # lora_dropout: the dropout probability for Lora layers
    # bias: either "none", "all", or "lora_only". Whose biases will be updated during training
    # target_modules: which specific modules to fine tune. Here updating only queries and values of the attention block
    
    # small reminder: queries -> 
    
    # you should use FEATURE_EXTRACTION for task_type for domain adapatation (measure difference between batches)
    
    # peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=kwargs['r'], lora_alpha=kwargs['la'],
    #                              lora_dropout=0.1, bias="lora_only", target_modules=["q", "v"])
    # model = get_peft_model(base_model, peft_config)
    
    return base_model


base_model = LoRA_model()

# Questions you should be able to answer before doing anything

# why did you use this base model? boh it was the same one used in DADER
# why are you upddating only the queries and the values
# lora_dropout, r, and lora_alpha are hyperparameters to tune