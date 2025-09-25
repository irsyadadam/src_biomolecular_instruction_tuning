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

import pandas as pd
import torch
import glob
import os

class ProteinPreprocess:
    def __init__(self, data_args=None):
        self.data_args = data_args
        self.proteomics_data = self._load_proteomics_data()
    
    def _load_proteomics_data(self):
        proteomics_dir = getattr(self.data_args, 'proteomics_data_path', 'data/proteomics')
        csv_files = glob.glob(os.path.join(proteomics_dir, '*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {proteomics_dir}")
        
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0)
            df.index = df.index.astype(str)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return combined_df
    
    def __call__(self, sample_id):
        if sample_id is None:
            return torch.zeros(self.proteomics_data.shape[1], dtype=torch.float32)
        
        sample_id = str(sample_id)
        if sample_id in self.proteomics_data.index:
            return torch.tensor(self.proteomics_data.loc[sample_id].values, dtype=torch.float32)
        else:
            return torch.zeros(self.proteomics_data.shape[1], dtype=torch.float32)

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
        vision_tower_type = getattr(data_args, 'vision_tower', 'mlp')
        if vision_tower_type in ['mlp', 'node_encoder']:
            self.protein_preprocess = ProteinPreprocess(data_args)
        #if graph tower
        else:
            self.protein_preprocess = None

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
            
            vision_tower_type = getattr(self.data_args, 'vision_tower')
            
            if vision_tower_type == 'graph_tower':
                data_dict['image'] = sample_id
            elif vision_tower_type == 'node_encoder':
                data_dict['image'] = sample_id  
            elif vision_tower_type == 'mlp':
                if self.protein_preprocess is not None:
                    protein_expression = self.protein_preprocess(sample_id)
                    data_dict['image'] = protein_expression
                else:
                    data_dict['image'] = torch.zeros(getattr(self.data_args, 'num_proteins', 4792))
                
        elif self.data_args.is_multimodal:
            if getattr(self.data_args, 'vision_tower', 'mlp') in ['node_encoder', 'graph_tower']:
                data_dict['image'] = None
            else:
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
            
            # Handle different types of images/data
            if images and images[0] is not None:
                # If string (node_encoder or graph_tower)
                if isinstance(images[0], str):
                    batch['images'] = images
                # If tensor (mlp)
                elif isinstance(images[0], torch.Tensor):
                    if all(x is not None and x.shape == images[0].shape for x in images):
                        batch['images'] = torch.stack(images)
                    else:
                        batch['images'] = images
                else:
                    batch['images'] = images
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