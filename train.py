from lib.proc_init_utils import initialise_tpu; initialise_tpu('v4-16', n_devices=1, rank=0)
import jax; print('Running on:', jax.numpy.zeros(()).device())
import jax.random as rand
import jax_smi
import optax
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer
from typing import Any, Callable
import wandb

from lib.seeding import BEST_INTEGER
from lib_new import DataLoader, TrainData, cross_entropy_loss, load_dataset

def load_model() -> tuple[Callable, dict]:
    config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
    model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
    forward = model.__call__
    params = model.params
    return forward, params

# import torch
# from transformers import T5ForConditionalGeneration
# config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
# model = T5ForConditionalGeneration.from_pretrained('base5', config=config)
# torch.equal(model.shared.weight, model.encoder.embed_tokens.weight)
# torch.equal(model.shared.weight, model.decoder.embed_tokens.weight)

# outputs = model.generate(seq, params=params, max_length=120, do_sample=True, top_k=0)
# tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

batch_size = 36
max_len = 128
n_epochs = 10

forward = optimize = ...  # dummy function

@jax.jit
@jax.value_and_grad
def train_forward(params: dict, data_batch: TrainData, *, key: rand.KeyArray):
    src, dst, mask_enc, mask_dec, labels = data_batch
    outputs = forward(
        input_ids=src,
        attention_mask=mask_enc,
        decoder_input_ids=dst,
        decoder_attention_mask=mask_dec,
        params=params,
        dropout_rng=key,
    )
    loss = cross_entropy_loss(outputs.logits, labels, mask=mask_dec)
    return loss

def train_step(params: dict, opt_state: Any, data_batch: TrainData, *, key: rand.KeyArray):
    loss, grads = train_forward(params, data_batch, key=key)
    updates, opt_state = optimize(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def main() -> None:
    global forward, optimize

    wandb.init(project='t5-finetuning-qa')
    jax_smi.initialise_tracking()

    tokenizer = T5Tokenizer.from_pretrained('base5')
    forward, params = load_model()

    dataset = load_dataset()
    dataloader = DataLoader(dataset, tokenizer, batch_size=batch_size, max_len=max_len)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(0.1, eps=0.001),
        optax.sgd(learning_rate=0.03),
    )
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    key = rand.PRNGKey(BEST_INTEGER)

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        for data_batch in dataloader:
            data_batch = jax.device_put(data_batch)
            key, subkey = rand.split(key)
            params, opt_state, loss = train_step(params, opt_state, data_batch, key=subkey)
            wandb.log({'train loss': loss}, commit=False)

    wandb.log({}, commit=True)

if __name__ == '__main__':
    main()
