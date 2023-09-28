import os; os.environ['CUDA_VISIABLE_DEVICES'] = '0'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
from pydantic import BaseModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Callable

from lib_translate import initialise_translator

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {'Hello': 'World'}

def load_chat_model() -> Callable:
    class InputData(BaseModel):
        history: list[str]
        sentence: str

    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = LlamaForCausalLM.from_pretrained('../llamax-qa-weights', torch_dtype=torch.float16).to('cuda')

    def chat(data: InputData) -> str:
        sentence = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data.sentence}

### Response:
"""
        inputs = tokenizer([sentence], return_tensors='pt')
        outputs = model.generate(input_ids=inputs.input_ids.to('cuda'), do_sample=True, max_length=256)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text.removeprefix(sentence)

    return chat

app.post('/chat/')(load_chat_model())

def load_translate_model() -> Callable:
    with open('env.json', encoding='utf-8') as f:
        o = json.load(f)
    deepl_apikey = o['deepl_apikey']
    openai_apikey = o['openai_apikey']

    translate_text = initialise_translator(deepl_apikey, openai_apikey)

    class InputData(BaseModel):
        sentence: str
        src_lang: str
        dst_lang: str

    def translate(data: InputData) -> str:
        sentence = data.sentence
        src_lang = data.src_lang
        dst_lang = data.dst_lang
        return translate_text(sentence, src_lang, dst_lang)
    
    return translate

app.post('/translate/')(load_translate_model())
