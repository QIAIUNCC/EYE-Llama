'''
The sft script is adapted from PMC-LLaMA. Minor Changes are made to use QLoRA and the template prompt for Llama2-7b-chat.
'''
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import os
from datetime import datetime

import torch
import transformers
# import utils
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import Trainer, BitsAndBytesConfig
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


peft_lora_r=16
peft_lora_alpha=64
target_modules =["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
        "<s>[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n{input} [/INST]"
    )
}



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    #cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    ) 


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            
        )
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    print("train dataset sample: ")
    print(examples[0])
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        
        logging.warning("Formatting inputs...")
        prompt_input = PROMPT_DICT["prompt_input"]
        sources = [
            prompt_input.format_map(example) for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Loading data...")
    #list_data_dict = utils.jload(data_path)
    list_data_dict = load_from_disk(data_args.data_path)
    train_data_dict = list_data_dict['train']
    eval_data_dict = list_data_dict['test']


    train_dataset = SupervisedDataset(train_data_dict, tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(eval_data_dict, tokenizer=tokenizer)
    print('the length of the training and validation sets:\n')
    print(len(train_dataset))
    print(len(eval_dataset))

    # raise RuntimeError(train_dataset[0])
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    time_str = datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    training_args.output_dir = os.path.join(training_args.output_dir, f"{training_args.run_name}-{time_str}")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )

    torch_dtype = torch.bfloat16

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )

    peft_config = LoraConfig(
    r=peft_lora_r,
    lora_alpha=peft_lora_alpha,
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=target_modules,
    )
    
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True}) 
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    print(type(model))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        #cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        #padding_side="right",
        use_fast=False,
    )
    #special_tokens_dict = dict()
    #if tokenizer.pad_token is None or tokenizer.pad_token == '':
    #    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    #if tokenizer.eos_token is None or tokenizer.eos_token == '':
    #    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    #if tokenizer.bos_token is None or tokenizer.bos_token == '':
    #    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    #if tokenizer.unk_token is None or tokenizer.unk_token == '':
    #    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    trainer.save_state()

    trainer.save_model(output_dir=training_args.output_dir)




if __name__ == "__main__":
    train()
