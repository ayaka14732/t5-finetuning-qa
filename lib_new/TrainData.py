from jax import Array
from typing import NamedTuple

class TrainData(NamedTuple):
    src: Array
    dst: Array
    src_mask: Array
    dst_mask: Array
    labels: Array
