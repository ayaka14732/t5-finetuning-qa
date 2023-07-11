import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
import jax_smi
import optax
import time
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer
from typing import Any, Callable, Optional
import wandb

from lib.proc_init_utils import initialise_tpu
from lib_new import MyDataLoader, TrainData, cross_entropy_loss, load_data

forward: Optional[Callable] = None
optimize: Optional[Callable] = None

def load_model() -> tuple[Callable, dict]:
    config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
    model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
    forward = model.__call__
    params = model.params
    return forward, params

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
    )  # type: ignore
    loss = cross_entropy_loss(outputs.logits, labels, mask=mask_dec)
    return loss

@jax.jit
def train_step(params: dict, opt_state: Any, total_loss: Array, data_batch: TrainData, key: rand.KeyArray):
    key, subkey = rand.split(key)
    loss, grads = train_forward(params, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, total_loss, loss, key

def main() -> None:
    global forward, optimize

    lr = 0.0025
    batch_size = 160
    max_len_enc = 256
    max_len_dec = 64
    n_epochs = 5
    rank = 2
    seed = 3407

    initialise_tpu('v4-16', n_devices=1, rank=rank)
    wandb.init(project='t5-finetuning-qa')
    jax_smi.initialise_tracking(rank=rank)
    key = rand.PRNGKey(seed)

    tokenizer = T5Tokenizer.from_pretrained('base5')
    forward, params = load_model()

    data = load_data()
    dataloader = MyDataLoader(data, tokenizer, batch_size, max_len_enc, max_len_dec)  # TODO: prng

    optimizer = optax.adafactor(learning_rate=lr)
    optimize = optimizer.update
    opt_state = optimizer.init(params)

    for epoch in range(n_epochs):
        total_loss = jnp.zeros(())
        for step, data_batch in enumerate(dataloader):
            start_time = time.time()
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, total_loss, data_batch, key)
            jax.debug.callback(lambda loss: wandb.log({'train loss': loss.item(), 'time': time.time() - start_time}), loss)
        wandb.log({'epoch loss': total_loss.item() / (step + 1)})

if __name__ == '__main__':
    main()
