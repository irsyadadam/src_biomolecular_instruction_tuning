import copy
from dataclasses import dataclass
import json
from typing import Dict, Sequence
import os
import torch
from torch.utils.data import Dataset
import transformers
import pandas as pd
import glob

from .text_preprocess import TextPreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *

class ProteinPreprocess:
    def __init__(self, data_args=None):
        self.data_args = data_args
        self.proteomics_data = self._load_proteomics_data()
    
    def _load_proteomics_data(self):
        proteomics_dir = getattr(self.data_args, 'proteomics_data_path', 'data/filtered_proteomics')
        proteomics_files = glob.glob(os.path.join(proteomics_dir, '*_filtered_proteomics.csv'))
        
        all_data = []
        for prot_file in proteomics_files:
            df = pd.read_csv(prot_file, index_col=0)
            all_data.append(df)
        
        return pd.concat(all_data, axis=0)
    
    def __call__(self, sample_id):
        if sample_id and sample_id in self.proteomics_data.index:
            protein_expression = torch.tensor(
                self.proteomics_data.loc[sample_id].values, 
                dtype=torch.float32
            )
        else:
            protein_expression = torch.zeros(self.proteomics_data.shape[1], dtype=torch.float32)
        
        return protein_expression

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        
        with open(data_path, "r") as f:
            if data_path.endswith('.jsonl'):
                list_data_dict = [json.loads(line) for line in f]
            else:
                list_data_dict = json.load(f)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.protein_preprocess = ProteinPreprocess(data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            protein_tokens = 1 if 'sample_id' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + protein_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'sample_id' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if 'conversations' in sources:
            conversations = sources['conversations']
        else:
            conversations = [
                {"from": "human", "value": f"<image>\n{sources['instruction']}"},
                {"from": "gpt", "value": sources['output']}
            ]
        
        data_dict = self.text_preprocess(copy.deepcopy(conversations))
        
        if 'sample_id' in sources:
            sample_id = sources['sample_id'][0] if isinstance(sources['sample_id'], list) else sources['sample_id']
            protein_expression = self.protein_preprocess(sample_id)
            data_dict['image'] = protein_expression
        elif self.data_args.is_multimodal:
            num_proteins = getattr(self.data_args, 'num_proteins', 4792)
            data_dict['image'] = torch.zeros(num_proteins)
        
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)