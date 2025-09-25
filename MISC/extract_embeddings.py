import torch
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.data.template.base import Template
from tinyllava.data.dataset_proteomics import ProteinPreprocess
from tinyllava.utils.arguments import DataArguments
import argparse

def load_model(checkpoint_path, device):
    is_lora = os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json'))
    
    if is_lora:
        from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
        from peft import PeftModel
        config = TinyLlavaConfig.from_pretrained(checkpoint_path)
        model = TinyLlavaForConditionalGeneration(config)
        model.load_llm(model_name_or_path=config.llm_model_name_or_path)
        model.load_vision_tower(model_name_or_path=config.vision_model_name_or_path)
        model.load_connector(pretrained_connector_path=os.path.join(checkpoint_path, 'connector'))
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        model = TinyLlavaForConditionalGeneration.from_pretrained(checkpoint_path)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    return model, tokenizer, model.config

def get_sample_ids(proteomics_data_path):
    import glob
    csv_files = glob.glob(os.path.join(proteomics_data_path, '*.csv'))
    all_ids = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, index_col=0)
        all_ids.extend(df.index.astype(str).tolist())
    return list(set(all_ids))

def extract_embeddings_batch(model, tokenizer, config, sample_ids, device, batch_size=32):
    vision_tower_type = getattr(config, 'vision_model_name_or_path', 'mlp')
    embeddings = []
    
    input_text = "<image>"
    input_ids = Template.tokenizer_image_token(input_text, tokenizer, return_tensors='pt')
    input_ids = input_ids.to(device)
    
    if vision_tower_type != 'node_encoder':
        data_args = DataArguments()
        data_args.proteomics_data_path = getattr(config, 'proteomics_data_path', '../biomolecule_instruction_tuning/data/filtered_proteomics/')
        data_args.num_proteins = getattr(config, 'num_proteins', 4792)
        protein_processor = ProteinPreprocess(data_args)
    
    for i in range(0, len(sample_ids), batch_size):
        batch_ids = sample_ids[i:i+batch_size]
        
        if vision_tower_type == 'node_encoder':
            proteomics_input = batch_ids
        else:
            batch_data = []
            for sample_id in batch_ids:
                data = protein_processor(sample_id)
                if data.sum() == 0:
                    data = torch.zeros_like(data)
                batch_data.append(data)
            proteomics_input = torch.stack(batch_data).to(device)
        
        batch_input_ids = input_ids.repeat(len(batch_ids), 1)
        
        with torch.no_grad():
            _, _, _, _, inputs_embeds, _ = model.prepare_inputs_labels_for_multimodal(
                batch_input_ids, None, None, None, None, proteomics_input
            )
            
            outputs = model.language_model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            final_hidden = outputs.hidden_states[-1]
            
            # Extract embeddings for proteomics tokens (skip text tokens)
            for j, sample_id in enumerate(batch_ids):
                sample_embedding = final_hidden[j].mean(dim=0).cpu().numpy()
                embeddings.append((sample_id, sample_embedding))
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i + len(batch_ids)}/{len(sample_ids)} samples")
    
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--proteomics_data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer, config = load_model(args.checkpoint_path, device)
    print(f"Model loaded. Vision tower: {getattr(config, 'vision_model_name_or_path', 'unknown')}")
    
    sample_ids = get_sample_ids(args.proteomics_data_path)
    print(f"Found {len(sample_ids)} samples")
    
    embeddings = extract_embeddings_batch(model, tokenizer, config, sample_ids, device, args.batch_size)
    
    # Convert to DataFrame
    embedding_matrix = np.stack([emb[1] for emb in embeddings])
    sample_names = [emb[0] for emb in embeddings]
    
    df = pd.DataFrame(embedding_matrix, index=sample_names)

    print(f"Output file: {args.output_path}")
    print(f"CSV shape: {df.shape}")
    print(f"Samples: {df.shape[0]}")
    print(f"Embedding dimensions: {df.shape[1]}")
    
    df.to_csv(args.output_path)
    print(f"Saved embeddings to {args.output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()