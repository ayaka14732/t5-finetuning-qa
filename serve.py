from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from itertools import chain
import jax
from jax import Array
import jax.random as rand
import json
from pydantic import BaseModel
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer
from typing import Callable

from lib.param_utils import load_params
from lib.proc_init_utils import initialise_gpu
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

    def make_prompt(history: list[str], sentence: str) -> str:
        return '\n'.join((
            'Generate a high-quality response for the conversation.',
            'Conversation:',
            *chain.from_iterable((
                'User: ' + history[i],
                'System: ' + history[i + 1]
            ) for i in range(0, len(history), 2)),
            'User: ' + sentence,
            'Response:',
        ))

    initialise_gpu(cuda_visible_devices='0')
    jax.experimental.compilation_cache.compilation_cache.initialize_cache('cache')

    key = rand.PRNGKey(3407)
    tokenizer = T5Tokenizer.from_pretrained('base5')
    config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
    model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
    model.params = load_params('worldly-lion-48.npy')

    @jax.jit
    def inference(input_ids: Array, attention_mask: Array, key: rand.KeyArray) -> Array:
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, max_length=64, prng_key=key)
        return output.sequences

    def chat(data: InputData) -> str:
        history = data.history
        sentence = data.sentence
        nonlocal key
        prompt = make_prompt(history, sentence)
        key, subkey = rand.split(key)
        inputs = tokenizer([prompt], padding='max_length', max_length=512, return_tensors='jax')
        outputs = inference(inputs.input_ids, inputs.attention_mask, subkey)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text
    
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
