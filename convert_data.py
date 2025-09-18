#!/usr/bin/env python3
"""
Convert biomolecular instruction tuning data to TinyLLaVA format
"""

import json
import os
from pathlib import Path

def convert_instruction_to_conversation(instruction_data):
    """
    Convert single instruction entry to TinyLLaVA conversation format
    
    Input format:
    {
        "instruction": "Analyze the proteomics data...",
        "input": "",
        "output": "The patient is predicted...",
        "sample_id": ["C3L-00104_tumor"]
    }
    
    Output format:
    {
        "sample_id": "C3L-00104_tumor",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nAnalyze the proteomics data..."
            },
            {
                "from": "gpt", 
                "value": "The patient is predicted..."
            }
        ]
    }
    """
    # Extract sample_id (handle both list and string formats)
    sample_id = instruction_data["sample_id"]
    if isinstance(sample_id, list):
        sample_id = sample_id[0]
    
    # Build the human message with <image> token
    human_message = "<image>\n" + instruction_data["instruction"]
    
    # Add input if it exists and is not empty
    if instruction_data.get("input", "").strip():
        human_message += "\n" + instruction_data["input"]
    
    # Create conversation format
    conversation_data = {
        "sample_id": sample_id,
        "conversations": [
            {
                "from": "human",
                "value": human_message
            },
            {
                "from": "gpt",
                "value": instruction_data["output"]
            }
        ]
    }
    
    return conversation_data

def convert_file(input_path, output_path):
    """Convert a single instruction file to TinyLLaVA format"""
    
    print(f"Converting {input_path} -> {output_path}")
    
    converted_data = []
    
    # Read input file (handle both .json and .jsonl)
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.suffix == '.jsonl':
            # JSONL format - one JSON object per line
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    instruction_data = json.loads(line)
                    conversation_data = convert_instruction_to_conversation(instruction_data)
                    converted_data.append(conversation_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        else:
            # JSON format - list of objects
            try:
                all_instructions = json.load(f)
                for instruction_data in all_instructions:
                    conversation_data = convert_instruction_to_conversation(instruction_data)
                    converted_data.append(conversation_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON file: {e}")
                return
    
    # Write output as JSON (TinyLLaVA expects .json format)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(converted_data)} samples")
    print(f"📁 Saved to: {output_path}")
    
    # Show sample output
    if converted_data:
        print(f"\n📋 Sample converted entry:")
        print(json.dumps(converted_data[0], indent=2))

def main():
    # Your data paths
    stage1_input = "/local/irsyadadam/biomolecular_instruction_tuning_data/instruct_generation/pretrain_instruction_pairs.jsonl"
    stage2_input = "/local/irsyadadam/biomolecular_instruction_tuning_data/instruct_generation/sft_instruction_pairs.jsonl"
    
    # Output paths
    output_dir = Path("/local/irsyadadam/biomolecular_instruction_tuning_data/final_data/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage1_output = output_dir / "proteomics_pretrain_conversations.json"
    stage2_output = output_dir / "proteomics_finetune_conversations.json"
    
    print("🧬 Converting Biomolecular Instruction Data to TinyLLaVA Format")
    print("=" * 70)
    
    # Convert Stage 1 (Pretrain)
    print("\n📊 Stage 1: Pretrain Data")
    if os.path.exists(stage1_input):
        convert_file(Path(stage1_input), stage1_output)
    else:
        print(f"❌ File not found: {stage1_input}")
    
    # Convert Stage 2 (SFT)  
    print("\n📊 Stage 2: Supervised Fine-tuning Data")
    if os.path.exists(stage2_input):
        convert_file(Path(stage2_input), stage2_output)
    else:
        print(f"❌ File not found: {stage2_input}")
    
    print(f"\n🎯 Conversion Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Stage 1: {stage1_output}")
    print(f"📄 Stage 2: {stage2_output}")
    
    print(f"\n💡 Next Steps:")
    print(f"1. Update your training script data paths:")
    print(f"   Stage 1: --data_path {stage1_output}")
    print(f"   Stage 2: --data_path {stage2_output}")
    print(f"2. Make sure your proteomics CSV files are in the proteomics_data_path")
    print(f"3. Ensure CSV files are named: {{sample_id}}_proteomics.csv")

if __name__ == "__main__":
    main()