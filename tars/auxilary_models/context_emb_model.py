import os
import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead
)
from tars.base.model import Model
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class ContextEmbeddingModel(Model):

    SEP_TOKEN = "[SEP]"

    def __init__(self, model_name_or_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelWithLMHead.from_pretrained(model_name_or_path)

        self.tokenizer.add_special_tokens({"additional_special_tokens": [SEP_TOKEN]})
        self.model.resize_token_embeddings(len(tokenizer))


    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                inputs: dict of padded tensors (output of data collator)
            Returns:
                last hidden state of model: tensor of shape [seq_len, hidden_dim]
        '''
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[0]


    def text_transforms(self, sents: List[str], is_goal: bool) -> Dict[str, Union[List[int], torch.Tensor]]:
        '''
            Args:
                sents: list of either the goal instruction or the low-level instructions
            Returns:
                inputs to contextual embedding model (to be futher processed by data collator)
        '''
        if is_goal:
            input_str = sents[0]
        else:
            input_str = f" {SEP_TOKEN} ".join(sents)
        result = self.tokenizer(input_str)
        return result


    def text_collate(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        '''
            Args:
                features: output of text_transforms
            Returns:
                batched inputs to contextual embedding model
        '''
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        return batch
        
        