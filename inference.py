import jax
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer

from lib.param_utils import load_params
from lib.proc_init_utils import initialise_tpu

rank = 3
initialise_tpu('v4-16', n_devices=1, rank=rank)

tokenizer = T5Tokenizer.from_pretrained('base5')
config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
params = load_params('prime-cherry-31.npy')

inputs = tokenizer(['How to go to the UK?', 'How can I sleep?'], padding='max_length', max_length=64, return_tensors='jax')
outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, params=params, do_sample=True, max_length=64)
results = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
print(results)
