import torch
from transformers import  AutoModelForCausalLM,LlamaTokenizer
import bitsandbytes
from distutils.version import LooseVersion
from peft import PeftModel

def load_peft_model(model,peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    #merged_model = peft_model.merge_and_unload()
    return peft_model 

def load_model():
    model = AutoModelForCausalLM.from_pretrained("../llama2_7b" ,load_in_4bit=True, device_map = {"": 0}, torch_dtype=torch.float16)
    tokenizer = LlamaTokenizer.from_pretrained("../llama2_7b")
    model = load_peft_model(model,"../llama-output") 
    return model,tokenizer

def generate(prompt,model,tokenizer) :
    eval_prompt = "### Question: "+prompt+"\n ### Answer:"
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        ch = tokenizer.decode(model.generate(**model_input, max_new_tokens=50 ,top_k=50, top_p=0.9, temperature=1.0, do_sample=True,repetition_penalty = 1.2 )[0], skip_special_tokens=True)
        return ch.split("### Answer:")[1].strip()
