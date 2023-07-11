import jax
import jax.random as rand
import jax_smi
import optax
from transformers import FlaxT5ForConditionalGeneration, T5Config, T5Tokenizer
from typing import Any, Callable, Optional
import wandb

from lib.proc_init_utils import initialise_tpu;  
from lib.seeding import BEST_INTEGER
from lib_new import MyDataLoader, TrainData, cross_entropy_loss, load_data

forward: Optional[Callable] = None
optimize: Optional[Callable] = None

def load_model() -> tuple[Callable, dict]:
    config = T5Config.from_pretrained('base5', tie_word_embeddings=False)
    model = FlaxT5ForConditionalGeneration.from_pretrained('base5', config=config, from_pt=True)
    forward = model.__call__
    params = model.params
    return forward, params

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
    )  # type: ignore
    loss = cross_entropy_loss(outputs.logits, labels, mask=mask_dec)
    return loss

def train_step(params: dict, opt_state: Any, data_batch: TrainData, *, key: rand.KeyArray):
    loss, grads = train_forward(params, data_batch, key=key)
    updates, opt_state = optimize(grads, opt_state, params)  # type: ignore
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def main() -> None:
    global forward, optimize

    batch_size = 72
    max_len = 128
    n_workers = 8
    n_epochs = 10

    rank = 2

    initialise_tpu('v4-16', n_devices=1, rank=rank)
    print('Running on:', jax.numpy.zeros(()).device())
    wandb.init(project='t5-finetuning-qa')
    if rank == 0:
        jax_smi.initialise_tracking()

    tokenizer = T5Tokenizer.from_pretrained('base5')
    forward, params = load_model()

    data = load_data()
    dataloader = MyDataLoader(data, tokenizer, batch_size, max_len, n_workers)  # TODO: prng

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
            print(f'Step')
            key, subkey = rand.split(key)
            params, opt_state, loss = train_step(params, opt_state, data_batch, key=subkey)
            wandb.log({'train loss': loss})

if __name__ == '__main__':
    main()
