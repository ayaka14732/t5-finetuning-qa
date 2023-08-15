import jax
from jax import Array
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from typing import Callable, Sequence

def shard_array_from_sharding_scheme(mesh: Sequence[int], name: Sequence[str]) -> Callable:
    devices = mesh_utils.create_device_mesh(mesh)
    mesh_ = Mesh(devices, axis_names=name)
    def inner(x: Array, spec: P) -> Array:
        shape = x.shape
        sharding = NamedSharding(mesh_, spec)
        xs = [jax.device_put(x[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
        return jax.make_array_from_single_device_arrays(shape, sharding, xs)
    return inner
