import torch
import os
from transformers import AutoTokenizer
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.data.template.base import Template
from tinyllava.data.dataset_proteomics import ProteinPreprocess
from tinyllava.utils.arguments import DataArguments

def load_model(checkpoint_path):
    """Load proteomics model from LoRA checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load config and create base model
    config = TinyLlavaConfig.from_pretrained(checkpoint_path)
    model = TinyLlavaForConditionalGeneration(config)
    
    # Load base components
    model.load_llm(model_name_or_path=config.llm_model_name_or_path)
    model.load_vision_tower(model_name_or_path=config.vision_model_name_or_path)
    model.load_connector(pretrained_connector_path=os.path.join(checkpoint_path, 'connector'))
    
    # Load LoRA adapter
    from peft import PeftModel
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    model.eval()
    print(f"Model loaded - Proteomics mode: {config.proteomics_mode}")
    return model, tokenizer, config

def test_inference(model, tokenizer, config, sample_id, question):
    """Run inference on a single sample"""
    # Load proteomics data
    data_args = DataArguments()
    data_args.proteomics_data_path = config.proteomics_data_path
    data_args.num_proteins = config.num_proteins
    
    protein_processor = ProteinPreprocess(data_args)
    proteomics_data = protein_processor(sample_id).unsqueeze(0).float()
    
    print(f"Proteomics data shape: {proteomics_data.shape}, sum: {proteomics_data.sum():.2f}")
    
    if proteomics_data.sum() == 0:
        return f"No data found for {sample_id}"
    
    # Prepare input
    input_text = f"<image>\n{question}"
    input_ids = Template.tokenizer_image_token(input_text, tokenizer, return_tensors='pt')
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    print(f"Input shape: {input_ids.shape}, tokens: {input_ids[0][:10].tolist()}...")
    
    # Generate response with better settings
    with torch.no_grad():
        outputs = model.generate(
            inputs=input_ids,
            images=proteomics_data,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    print(f"Output shape: {outputs.shape}")
    print(f"Full output tokens: {outputs[0].tolist()}")
    
    # Decode full output for debugging
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)    
    print(f"Full response: '{full_response}'")


def run_tests():
    """Run all tests"""
    model_path = "/local/irsyadadam/biomolecular_instruction_tuning_data/mlp_llm/finetune"
    
    # Load model once
    model, tokenizer, config = load_model(model_path)
    
    print(f"Model config:")
    print(f"\tProteomics mode: {getattr(config, 'proteomics_mode', False)}")
    print(f"\tNum proteins: {getattr(config, 'num_proteins', 'unknown')}")
    print(f"\tMLP tower type: {getattr(config, 'mlp_tower_type', 'unknown')}")
    print(f"\tVision tower: {getattr(config, 'vision_model_name_or_path', 'unknown')}")
    print(f"\tVocab size: {getattr(config, 'vocab_size', 'unknown')}")
    print(f"\tPad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"\tEOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    
    test_cases = [
        ("C3L-00104_tumor", "Analyze the proteomics data to infer the patient's prognosis."),
        ("C3L-00104_tumor", "What is the tumor size?"),
        ("C3L-00365_tumor", "What is the tumor classification?"),
    ]
    
    print("\nTesting PPI-graph LLM for Proteomics")
    print("=" * 60)
    
    for i, (sample_id, question) in enumerate(test_cases, 1):
        print(f"\n[{i}] Sample: {sample_id}")
        print(f"Q: {question}")
        
        try:
            response = test_inference(model, tokenizer, config, sample_id, question)
            print(f"A: {response}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_tests()