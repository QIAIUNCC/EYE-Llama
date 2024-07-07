
path_to_checkpoint = "path/to/peft/checkpoint"
path_to_merged_model = "path/to/merged/model"

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)

peft_model = PeftModel.from_pretrained(model, path_to_checkpoint)

# Merge the PEFT model with the base model
merged_model = peft_model.merge_and_unload()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint)

# Save the tokenizer
tokenizer.save_pretrained(path_to_merged_model)


# Save the merged model
merged_model.save_pretrained(path_to_merged_model)
