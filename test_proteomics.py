import torch
import os
from transformers import AutoTokenizer
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.utils.constants import IMAGE_TOKEN_INDEX
from tinyllava.data.template.base import Template
from tinyllava.data.dataset_proteomics import ProteinPreprocess
from tinyllava.utils.arguments import DataArguments
from peft import PeftModel

def load_proteomics_model(model_path):
    """Load the trained proteomics model using standard approach"""
    print(f"Loading model from: {model_path}")
    
    try:
        # Use standard loading first (recommended)
        model = TinyLlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        config = model.config
        
        # Load tokenizer - use the one from config, not manual path
        tokenizer = model.tokenizer
        
        # Verify proteomics configuration
        if not getattr(config, 'proteomics_mode', False):
            print("⚠️ Warning: Model not in proteomics mode, but continuing...")
        
        model.eval()
        print("✅ Successfully loaded model using standard approach")
        print(f"Config - Proteomics mode: {getattr(config, 'proteomics_mode', False)}")
        print(f"Config - Num proteins: {getattr(config, 'num_proteins', 'Not set')}")
        
        return model, tokenizer, config
        
    except Exception as e:
        print(f"❌ Standard loading failed: {e}")
        print("Trying manual loading approach...")
        return load_lora_model_manual(model_path)

def load_lora_model_manual(model_path):
    """Fallback: Manual loading approach"""
    print(f"Manual loading from: {model_path}")
    
    # Load config
    if os.path.exists(os.path.join(model_path, "config.json")):
        config = TinyLlavaConfig.from_pretrained(model_path)
    else:
        raise FileNotFoundError(f"No config.json found in {model_path}")
    
    # Ensure proteomics mode is enabled
    if not getattr(config, 'proteomics_mode', False):
        print("⚠️ Enabling proteomics mode manually...")
        config.proteomics_mode = True
        config.num_proteins = 4792
        config.mlp_tower_type = 'mlp_3'
        config.mlp_hidden_size = 256
        config.mlp_dropout = 0.3
        config.proteomics_data_path = "../biomolecule_instruction_tuning/data/filtered_proteomics/"
    
    # Create model
    model = TinyLlavaForConditionalGeneration(config)
    
    # Load weights
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_model_name_or_path,
        use_fast=False,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    model.eval()
    return model, tokenizer, config

def test_proteomics_model(model_path, sample_id, question):
    """Test the proteomics model with proper handling"""
    
    try:
        # Load model and tokenizer
        model, tokenizer, config = load_proteomics_model(model_path)
        
        # Setup data args for proteomics
        data_args = DataArguments()
        data_args.proteomics_data_path = getattr(config, 'proteomics_data_path', 
                                                 "../biomolecule_instruction_tuning/data/filtered_proteomics/")
        data_args.num_proteins = getattr(config, 'num_proteins', 4792)
        
        # Load proteomics data for the patient
        protein_processor = ProteinPreprocess(data_args)
        proteomics_tensor = protein_processor(sample_id)
        
        if proteomics_tensor.sum() == 0:
            print(f"⚠️ Warning: All zeros for sample {sample_id}")
            return f"No proteomics data found for sample {sample_id}"
        
        proteomics_tensor = proteomics_tensor.unsqueeze(0).float()  # Add batch dimension
        
        # Prepare input text with image token
        input_text = f"<image>\n{question}"
        print(f"Input text: {input_text}")
        
        # Tokenize using the template method
        input_tokens = Template.tokenizer_image_token(input_text, tokenizer, return_tensors='pt')

        # Ensure input_tokens has batch dimension
        if input_tokens.dim() == 1:
            input_tokens = input_tokens.unsqueeze(0)
        
        print(f"Input tokens shape: {input_tokens.shape}")
        print(f"Proteomics tensor shape: {proteomics_tensor.shape}")
        print(f"Contains IMAGE_TOKEN_INDEX (-200): {(input_tokens == IMAGE_TOKEN_INDEX).any()}")
        
        # Move to appropriate device
        device = next(model.parameters()).device
        input_tokens = input_tokens.to(device)
        proteomics_tensor = proteomics_tensor.to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_tokens,
                images=proteomics_tensor,  # This should work in proteomics mode
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Handle decoding - only decode the newly generated tokens
        input_length = input_tokens.shape[-1]
        new_tokens = outputs[0][input_length:]
        
        print(f"Generated {len(new_tokens)} new tokens")
        
        # Decode only the new tokens
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_multiple_samples(model_path, test_cases):
    """Test multiple samples and questions"""
    
    results = []
    
    for i, (sample_id, question, expected_type) in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {sample_id} ---")
        print(f"Question: {question}")
        print(f"Expected type: {expected_type}")
        
        output = test_proteomics_model(model_path, sample_id, question)
        
        if output:
            print(f"Generated: {output}")
            results.append({
                "sample_id": sample_id,
                "question": question,
                "expected": expected_type,
                "generated": output,
                "status": "success"
            })
        else:
            print("Failed to generate response")
            results.append({
                "sample_id": sample_id,
                "question": question,
                "expected": expected_type,
                "generated": None,
                "status": "failed"
            })
        
        print("-" * 60)
    
    return results

def analyze_results(results):
    """Analyze the test results"""
    
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = total - successful
    
    print(f"\n=== Results Summary ===")
    print(f"Total tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    if successful > 0:
        print(f"\nSample successful outputs:")
        for r in results[:3]:  # Show first 3 successful
            if r["status"] == "success":
                print(f"Q: {r['question']}")
                print(f"A: {r['generated']}")
                print()

# Main test script
if __name__ == "__main__":
    # Update this path to your actual model checkpoint
    model_path = "/local/irsyadadam/biomolecular_instruction_tuning_data/finetune_output"
    
    # Test cases: (sample_id, question, expected_type)
    test_cases = [
        ("C3L-00104_tumor", "Analyze the proteomics data to infer the patient's prognosis.", "survival"),
        ("C3L-00104_tumor", "Determine the tumor size utilizing the provided pattern of protein expression.", "size"),
        ("C3L-00104_tumor", "Based on the protein expression profile, what is the histologic grade indicated by this molecular signature?", "grade"),
        ("C3L-00365_tumor", "Utilizing the proteomics profile of this patient, identify the corresponding tumor classification.", "classification"),
        ("C3L-00104_tumor", "Considering the molecular profile of this patient, would administering adjuvant radiation therapy be warranted?", "treatment"),
    ]
    
    print("Testing PPI-graph LLM for Proteomics...")
    print("=" * 70)
    
    # Run tests
    results = test_multiple_samples(model_path, test_cases)
    
    # Analyze results
    analyze_results(results)
    
    print(f"\nTesting complete!")
    print(f"Your proteomics LLM baseline is ready for evaluation!")