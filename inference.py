import jax
import jax.random as rand
from itertools import chain
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer

from lib.param_utils import load_params
from lib.proc_init_utils import initialise_gpu

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

rank = 3
initialise_gpu(cuda_visible_devices='0')
jax.experimental.compilation_cache.compilation_cache.initialize_cache('cache')

tokenizer = T5Tokenizer.from_pretrained('base5')
config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
model.params = load_params('worldly-lion-48.npy')

@jax.jit
def inference(input_ids, attention_mask, key):
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, max_length=64, prng_key=key)
    return output.sequences

history: list[str] = []

# How can I go from Tur Abdin to Stockholm?
# I want to book a train. List all Tuesday departures to Cambridge.
key = rand.PRNGKey(3407)

while True:
    sentence = input('>>> ')

    if sentence == '.':
        history = []
        continue
    elif sentence == ',':
        break

    prompt = make_prompt(history, sentence)
    # print('Using prompt:', prompt)

    key, subkey = rand.split(key)
    inputs = tokenizer([prompt], padding='max_length', max_length=512, return_tensors='jax')
    outputs = inference(inputs.input_ids, inputs.attention_mask, key)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(generated_text)

    history.append(sentence)
    history.append(generated_text)
