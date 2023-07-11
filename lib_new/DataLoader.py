import jax
from jax import Array
import jax.numpy as jnp
import random
from transformers import PreTrainedTokenizer
from typing import Any, NamedTuple

def chunks(lst: list[Any], *, chunk_size: int) -> list[list[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

class TrainData(NamedTuple):
    src: Array
    dst: Array
    src_mask: Array
    dst_mask: Array
    labels: Array

class DataLoader:
    def __init__(self, dataset: list[tuple[str, str]], tokenizer: PreTrainedTokenizer, *, batch_size: int, max_len: int, shuffle: bool=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            dataset = self.dataset[:]
            random.shuffle(dataset)  # TODO: key

        cpu_device = jax.devices('cpu')[0]

        with jax.default_device(cpu_device):
            for data_batch in chunks(dataset, chunk_size=self.batch_size):  # TODO: trailing stray values
                seq_src, seq_dst = zip(*data_batch)
                inputs_src = self.tokenizer(seq_src, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='jax')
                inputs_dst = self.tokenizer(seq_dst, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='jax')

                src = inputs_src.input_ids.astype(jnp.uint16)
                dst = inputs_dst.input_ids.astype(jnp.uint16)
                src_mask = inputs_src.attention_mask.astype(jnp.bool_)
                dst_mask = inputs_dst.attention_mask.astype(jnp.bool_)
                labels = jnp.roll(dst, -1, axis=-1).at[:, -1].set(0)

                yield TrainData(src, dst, src_mask, dst_mask, labels)
