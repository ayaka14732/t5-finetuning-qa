from functools import partial
import jax.numpy as jnp
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

from .TrainData import TrainData
from lib.proc_init_utils import initialise_cpu

class MyDataset(Dataset):
    def __init__(self, data: list[tuple[str, str]]) -> None:
        self.data = data
        super().__init__()

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

def transform(tokenizer: T5Tokenizer, max_len_enc: int, max_len_dec: int, data_batch: list[tuple[str, str]]) -> TrainData:
    seq_src, seq_dst = zip(*data_batch)
    inputs_src = tokenizer(seq_src, padding='max_length', max_length=max_len_enc, truncation=True, return_tensors='jax')
    inputs_dst = tokenizer(seq_dst, padding='max_length', max_length=max_len_dec, truncation=True, return_tensors='jax')

    src = inputs_src.input_ids.astype(jnp.uint16)
    dst = inputs_dst.input_ids.astype(jnp.uint16)
    src_mask = inputs_src.attention_mask.astype(jnp.bool_)
    dst_mask = inputs_dst.attention_mask.astype(jnp.bool_)
    labels = jnp.roll(dst, -1, axis=-1).at[:, -1].set(0)

    return TrainData(src, dst, src_mask, dst_mask, labels)

def worker_init_fn(*args, **kwargs) -> None:  # ignore all parameters
    initialise_cpu()

class MyDataLoader(DataLoader):
    def __init__(self, data: list[tuple[str, str]], tokenizer: T5Tokenizer, batch_size: int, max_len_enc: int, max_len_dec: int) -> None:
        dataset = MyDataset(data)
        collate_fn = partial(transform, tokenizer, max_len_enc, max_len_dec)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            prefetch_factor=1,
            multiprocessing_context=multiprocessing.get_context('spawn'),
            persistent_workers=True,
        )
