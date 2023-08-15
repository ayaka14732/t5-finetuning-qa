import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from jax.sharding import PartitionSpec as P
import jax_smi
import optax
import time
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer
from typing import Any, Callable, Optional
import wandb

from lib.multihost_utils import shard_array_from_sharding_scheme
from lib.param_utils import save_params
from lib.proc_init_utils import initialise_tpu
from lib_new import MyDataLoader, TrainData, cross_entropy_loss, load_data

forward: Optional[Callable] = None
optimize: Optional[Callable] = None
S: Optional[Callable] = None

def load_model() -> tuple[Callable, dict]:
    with jax.default_device(jax.devices('cpu')[0]):
        config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
        model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
        forward = model.__call__
        params = model.params
    params = jax.tree_map(lambda x: S(x, P(None,)), params)  # type: ignore
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
def train_step(params: dict, opt_state: Any, total_loss: Array, data_batch: TrainData, key: rand.KeyArray) -> tuple[dict, Any, Array, Array, rand.KeyArray]:
    key, subkey = rand.split(key)
    loss, grads = train_forward(params, data_batch, key=subkey)
    total_loss += loss
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, total_loss, loss, key

def main() -> None:
    global forward, optimize, S

    lr = 0.0023
    batch_size = 56
    max_len_enc = 512
    max_len_dec = 64
    n_epochs = 8
    seed = 3407

    initialise_tpu('v3-256', n_devices=8)
    wandb.init(project='t5-finetuning-qa')
    print(wandb.run.name)  # type: ignore
    jax_smi.initialise_tracking()
    key = rand.PRNGKey(seed)

    S = shard_array_from_sharding_scheme((8,), ('B',))

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
            data_batch = jax.tree_map(lambda x: S(x, P('B',)), data_batch)  # type: ignore
            params, opt_state, total_loss, loss, key = train_step(params, opt_state, total_loss, data_batch, key)
            jax.debug.callback(lambda loss: wandb.log({'train loss': loss.item(), 'time': time.time() - start_time}), loss)
        wandb.log({'epoch loss': total_loss.item() / (step + 1)})

    save_params(params, f'{wandb.run.name}.npy')  # type: ignore

if __name__ == '__main__':
    main()
