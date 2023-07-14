import evaluate
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as rand
from tqdm import tqdm
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer

from lib.param_utils import load_params
from lib.proc_init_utils import initialise_tpu
from lib.seeding import HASHED_BUDDHA
from lib_new import MyDataLoader, load_data

model = tokenizer = None  # type: ignore

def collate_fn(tokenizer: T5Tokenizer, max_len_enc: int, data_batch: list[tuple[str, str]]):
    seq_src, seq_dst = zip(*data_batch)
    inputs_src = tokenizer(seq_src, padding='max_length', max_length=max_len_enc, truncation=True, return_tensors='jax')

    src = inputs_src.input_ids.astype(jnp.uint16)
    src_mask = inputs_src.attention_mask.astype(jnp.bool_)

    return src, src_mask, seq_dst

@jax.jit
def inference(src, src_mask, key):
    output = model.generate(input_ids=src, attention_mask=src_mask, do_sample=True, max_length=64, prng_key=key)
    return output.sequences

def do_inference(src, src_mask, key):
    output = inference(src, src_mask, key)
    predictions = tokenizer.batch_decode(output, skip_special_tokens=True)
    return predictions

def main() -> None:
    global model, tokenizer

    rank = 0
    initialise_tpu('v4-16', n_devices=1, rank=rank)

    tokenizer = T5Tokenizer.from_pretrained('base5')
    config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
    model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
    model.params = load_params('fallen-thunder-44.npy')

    key = rand.PRNGKey(HASHED_BUDDHA)
    data = load_data(split='test')

    max_len_enc = 512
    max_len_dec = 64

    collate_fn_ = partial(collate_fn, tokenizer, max_len_enc)
    dataloader = MyDataLoader(data=data, tokenizer=tokenizer, batch_size=256, max_len_enc=max_len_enc, max_len_dec=max_len_dec, drop_last=False, collate_fn=collate_fn_)

    predictions = []
    references = []

    for src, src_mask, seq_dst in tqdm(dataloader):
        key, subkey = rand.split(key)
        y_hat = do_inference(src, src_mask, subkey)
        predictions.extend(y_hat)
        references.extend(seq_dst)

    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=predictions, references=[[x] for x in references])
    print(results)

    with open('1.txt', 'w', encoding='utf-8') as f:
        for a, b in zip(predictions, references):
            print(a, b, sep='\t', file=f)

if __name__ == '__main__':
    main()
