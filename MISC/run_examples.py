import torch
import os
from transformers import AutoTokenizer
from tinyllava.model.modeling_tinyllava import TinyLlavaForConditionalGeneration
from tinyllava.model.configuration_tinyllava import TinyLlavaConfig
from tinyllava.data.template.base import Template
from tinyllava.data.dataset_proteomics import ProteinPreprocess
from tinyllava.utils.arguments import DataArguments

def load_model(checkpoint_path):
    """Load proteomics model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Check if it's a LoRA checkpoint
    is_lora = os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json'))
    
    if is_lora:
        # Load LoRA model
        print("Loading LoRA checkpoint...")
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
    else:
        # Load regular checkpoint
        print("Loading regular checkpoint...")
        model = TinyLlavaForConditionalGeneration.from_pretrained(checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    model.eval()
    print(f"Model loaded - Proteomics mode: {getattr(model.config, 'proteomics_mode', False)}")
    return model, tokenizer, model.config

def get_available_sample_ids(model):
    """Get available sample IDs from the node encoder"""
    try:
        if hasattr(model.vision_tower._vision_tower, 'sample_id_to_node_idx'):
            available_ids = list(model.vision_tower._vision_tower.sample_id_to_node_idx.keys())
            print(f"Found {len(available_ids)} available sample IDs in graph")
            print(f"First 10 sample IDs: {available_ids[:10]}")
            return available_ids
        else:
            print("Node encoder not properly initialized or no sample mapping found")
            return []
    except Exception as e:
        print(f"Error getting sample IDs: {e}")
        return []

def test_inference(model, tokenizer, config, sample_id, question):
    """Run inference on a single sample"""
    print(f"\n=== Testing inference for {sample_id} ===")
    
    try:
        # Handle different vision tower types
        vision_tower_type = getattr(config, 'vision_model_name_or_path', 'mlp')
        
        if vision_tower_type == 'node_encoder':
            # For node encoder, pass the sample ID as string (or list for batch)
            proteomics_input = [sample_id]  # Pass as list for batching
            print(f"Using node encoder with sample ID: {sample_id}")
        else:
            # For MLP tower, load the actual proteomics data
            data_args = DataArguments()
            data_args.proteomics_data_path = getattr(config, 'proteomics_data_path', '../biomolecule_instruction_tuning/data/filtered_proteomics/')
            data_args.num_proteins = getattr(config, 'num_proteins', 4792)
            
            protein_processor = ProteinPreprocess(data_args)
            proteomics_data = protein_processor(sample_id)
            
            print(f"Proteomics data shape: {proteomics_data.shape}, sum: {proteomics_data.sum():.2f}")
            
            if proteomics_data.sum() == 0:
                return f"No data found for {sample_id}"
            
            # Add batch dimension for MLP
            proteomics_input = proteomics_data.unsqueeze(0).float()
        
        # Prepare input text
        input_text = f"<image>\n{question}"
        input_ids = Template.tokenizer_image_token(input_text, tokenizer, return_tensors='pt')
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Input tokens: {input_ids[0][:10].tolist()}")
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Only move to device if it's a tensor
        if hasattr(proteomics_input, 'to'):
            proteomics_input = proteomics_input.to(device)
                # Debug: Print proteomics input shape before generation
        print(f"[DEBUG] Proteomics input type: {type(proteomics_input)}")
        if hasattr(proteomics_input, 'shape'):
            print(f"[DEBUG] Proteomics input shape: {proteomics_input.shape}")
            print(f"[DEBUG] Proteomics input dtype: {proteomics_input.dtype}")
            print(f"[DEBUG] Proteomics input sum: {proteomics_input.sum():.4f}")
        else:
            print(f"[DEBUG] Proteomics input content: {proteomics_input}")

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs=input_ids,
                images=proteomics_input,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        print(f"Output shape: {outputs.shape}")
        
        # Decode only the generated part (skip input tokens)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)    
        print(f"Full response: '{full_response}'")
        
        return full_response
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

def run_tests():
    """Run all tests"""
    model_path = "/local/irsyadadam/biomolecular_instruction_tuning_data/mlp_llm/finetune"
    
    print("=== TinyLLaVA Proteomics Model Testing ===")
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        print("Please check the path and ensure the model has been trained.")
        return
    
    try:
        # Load model
        model, tokenizer, config = load_model(model_path)
        
        print(f"\n=== Model Configuration ===")
        print(f"Proteomics mode: {getattr(config, 'proteomics_mode', False)}")
        print(f"Num proteins: {getattr(config, 'num_proteins', 'unknown')}")
        print(f"Vision tower: {getattr(config, 'vision_model_name_or_path', 'unknown')}")
        
        if hasattr(config, 'mlp_tower_type'):
            print(f"MLP tower type: {config.mlp_tower_type}")
            print(f"MLP hidden size: {getattr(config, 'mlp_hidden_size', 'unknown')}")
        
        if hasattr(config, 'node_tower_type'):
            print(f"Node tower type: {config.node_tower_type}")
            print(f"Node hidden size: {getattr(config, 'node_hidden_size', 'unknown')}")
        
        print(f"Vocab size: {getattr(config, 'vocab_size', 'unknown')}")
        print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        
        # Test cases
        test_cases = [
            ("C3L-00104_tumor", "Analyze the proteomics data to infer the patient's prognosis."),
            ("C3L-00104_tumor", "What is the tumor size?"),
            ("C3L-00365_tumor", "What is the tumor classification?"),
            ("C3L-00104_tumor", "Determine the patient's sex based on the molecular profile."),
            ("C3L-00365_tumor", "What treatment would you recommend based on this proteomics profile?"),
        ]
        
        print("\n=== Running Inference Tests ===")
        
        for i, (sample_id, question) in enumerate(test_cases, 1):
            print(f"\n[Test {i}/5] Sample: {sample_id}")
            print(f"Question: {question}")
            
            try:
                response = test_inference(model, tokenizer, config, sample_id, question)
                print(f"Response: {response}")
                
            except Exception as e:
                print(f"Test failed with error: {e}")
                import traceback
                traceback.print_exc()
            
            print("-" * 60)
    
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()