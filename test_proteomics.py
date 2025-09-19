import torch
import os
from transformers import AutoTokenizer
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX
from tinyllava.data.template.base import Template
from tinyllava.data.dataset_proteomics import ProteinPreprocess
from tinyllava.utils.arguments import DataArguments

def load_deepspeed_checkpoint(checkpoint_dir = "/local/irsyadadam/biomolecular_instruction_tuning_data/mlp_llm/pretrain"):
    """Convert DeepSpeed checkpoint to standard model"""
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        raise FileNotFoundError("No DeepSpeed checkpoints found")
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading DeepSpeed checkpoint: {latest_checkpoint}")
    return load_state_dict_from_zero_checkpoint(checkpoint_path)

def load_proteomics_model(model_path):
    """Load proteomics model with DeepSpeed support"""
    print(f"Loading model from: {model_path}")
    
    config = TinyLlavaConfig.from_pretrained(model_path)
    
    if not getattr(config, 'proteomics_mode', False):
        config.proteomics_mode = True
        config.num_proteins = 4792
        config.mlp_tower_type = 'mlp_3'
        config.mlp_hidden_size = 256
        config.mlp_dropout = 0.3
        config.proteomics_data_path = "../biomolecule_instruction_tuning/data/filtered_proteomics/"
    
    model = TinyLlavaForConditionalGeneration(config)
    
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
    else:
        state_dict = load_deepspeed_checkpoint(model_path)
    
    model.load_state_dict(state_dict, strict=False)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_name_or_path,
        use_fast=False,
        padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    model.eval()
    print(f"✅ Model loaded - Proteomics mode: {config.proteomics_mode}")
    return model, tokenizer, config

def test_proteomics_model(model_path, sample_id, question):
    """Test proteomics model inference"""
    model, tokenizer, config = load_proteomics_model(model_path)
    
    data_args = DataArguments()
    data_args.proteomics_data_path = getattr(config, 'proteomics_data_path', 
                                             "../biomolecule_instruction_tuning/data/filtered_proteomics/")
    data_args.num_proteins = getattr(config, 'num_proteins', 4792)
    
    protein_processor = ProteinPreprocess(data_args)
    proteomics_tensor = protein_processor(sample_id).unsqueeze(0).float()
    
    if proteomics_tensor.sum() == 0:
        return f"No proteomics data found for sample {sample_id}"
    
    input_text = f"<image>\n{question}"
    input_tokens = Template.tokenizer_image_token(input_text, tokenizer, return_tensors='pt')
    
    if input_tokens.dim() == 1:
        input_tokens = input_tokens.unsqueeze(0)
    
    device = next(model.parameters()).device
    input_tokens = input_tokens.to(device)
    proteomics_tensor = proteomics_tensor.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs=input_tokens,
            images=proteomics_tensor,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    input_length = input_tokens.shape[-1]
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response.strip()

def run_tests():
    """Run proteomics model tests"""
    model_path = "/local/irsyadadam/biomolecular_instruction_tuning_data/mlp_llm/pretrain"
    
    test_cases = [
        ("C3L-00104_tumor", "Analyze the proteomics data to infer the patient's prognosis.", "survival"),
        ("C3L-00104_tumor", "Determine the tumor size utilizing the provided pattern of protein expression.", "size"),
        ("C3L-00104_tumor", "Based on the protein expression profile, what is the histologic grade indicated by this molecular signature?", "grade"),
        ("C3L-00365_tumor", "Utilizing the proteomics profile of this patient, identify the corresponding tumor classification.", "classification"),
        ("C3L-00104_tumor", "Considering the molecular profile of this patient, would administering adjuvant radiation therapy be warranted?", "treatment"),
    ]
    
    print("🧬 Testing PPI-graph LLM for Proteomics")
    print("=" * 50)
    
    results = []
    for i, (sample_id, question, expected_type) in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {sample_id} - {expected_type}")
        
        try:
            output = test_proteomics_model(model_path, sample_id, question)
            print(f"Q: {question}")
            print(f"A: {output}")
            results.append({"status": "success", "output": output})
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"status": "failed", "error": str(e)})
    
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"\n✅ Results: {successful}/{len(test_cases)} successful")
    return results

if __name__ == "__main__":
    run_tests()