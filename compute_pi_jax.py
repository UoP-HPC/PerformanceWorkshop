#with HPC python
from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnums=(1,))
def compute_pi_jax(key, Niter):

    key1, key2 = jax.random.split(key)

    x = jax.random.uniform(key1, (Niter,))
    y = jax.random.uniform(key2, (Niter,))

    inside = (x*x + y*y) < 1.0

    count = jnp.sum(inside)

    return 4.0 * count / Niter

#warming up
key = jax.random.PRNGKey(42)
compute_pi_jax(key, Niter).block_until_ready()


