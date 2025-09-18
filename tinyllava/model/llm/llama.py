from transformers import LlamaForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('llama')
def return_llamaclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return (LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load))

@register_llm('vicuna')  # Add this registration
def return_vicunaclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token
        return tokenizer
    return (LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load))