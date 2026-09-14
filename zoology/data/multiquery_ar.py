
import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

#: Bumped 2026-08-02 when the generator moved off the global numpy/torch RNGs onto a per-segment
#: `np.random.default_rng(seed)` stream (see `multiquery_ar` docstring, "DETERMINISM"). This field
#: is a plain (non-private) pydantic field, so it participates in `model_dump()` and therefore in
#: `DataSegment.from_config`'s on-disk cache-key hash (`zoology/data/utils.py`) — bumping it
#: invalidates every stale cache written by the pre-fix generator without deleting anyone's files.
MQAR_GEN_VERSION = 2


class MQARConfig(DataSegmentConfig):
    name: str="multiquery_ar"
    power_a: float=0.01
    num_kv_pairs: int=8
    random_non_queries: bool=True
    include_slices: bool=True
    gen_version: int = MQAR_GEN_VERSION

    def build(self, seed: int) -> DataSegment:
        return multiquery_ar(**self.model_dump(), seed=seed)


def _distinct_per_row(rng, n_rows, universe, k, chunk=8192):
    """`[n_rows, k]` integers in `[0, universe)`, distinct WITHIN a row, uniform.

    Mirrors `zoology.data.cue_consistency._distinct_per_row` exactly (same argsort-of-random-keys
    identity for a per-row `np.random.choice(replace=False)`), chunked so the transient key matrix
    stays bounded."""
    if k > universe:
        raise ValueError(f"cannot draw {k} distinct values from a universe of {universe}")
    out = np.empty((n_rows, k), dtype=np.int64)
    for lo in range(0, n_rows, chunk):
        hi = min(lo + chunk, n_rows)
        out[lo:hi] = np.argsort(rng.random((hi - lo, universe)), axis=1)[:, :k]
    return out


def _weighted_distinct(rng, n_rows, weights, k):
    """Weighted sampling WITHOUT replacement, `k` per row, over `len(weights)` positions.

    Mirrors `zoology.data.cue_consistency._weighted_distinct`: the Gumbel-top-k identity
    reproduces `np.random.choice(p=weights, replace=False)` exactly, vectorized over rows."""
    g = -np.log(-np.log(rng.random((n_rows, len(weights)))))
    return np.argsort(-(np.log(weights)[None, :] + g), axis=1)[:, :k]

def multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    num_passes: int=1,
    random_non_queries: bool=True,
    include_slices: bool=True,
    **kwargs
) -> DataSegment:
    """
    Generates synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    Example: 
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0 
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.
    
    We include one important note on the power law distribution. In real language data, 
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again 
    drops the further away from the orginal mention we are. In our synthetic, we can 
    control this with the power law parameters `train_power_a` and `test_power_a`. 
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01  
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
            paper, large vocabulary sizes (>1k) can be important for highlighting 
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults 
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to 
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In 
            In Figure 2 of the Zoology paper, we vary the input sequence length from 
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of unique key-value pairs in the sequence. 
        num_passes (int): The number of passes through the key-value pairs in the sequence
            This follows the JRT approach (https://arxiv.org/abs/2407.05483)
        train_power_a (float, optional): The power for the power law distribution for 
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for 
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.

    DETERMINISM (fixed 2026-08-02). Every draw below comes from a `np.random.default_rng(seed)`
    stream local to this call — never `np.random.seed` (global numpy) and never `torch.randint`
    without an explicit `torch.Generator` (global torch). The pre-fix version did both: it called
    `np.random.seed(seed)` and then read the resulting GLOBAL numpy stream via
    `np.random.choice`/`np.apply_along_axis`, and it filled `random_non_queries` positions with
    `torch.randint(...)` against torch's GLOBAL RNG with no seed of its own at all. Two consequences
    followed: (1) a segment's content could be perturbed by anything else in the process that had
    already touched either global stream before this call ran, and (2) the filler content was not
    reproducible from `seed` alone. See `zoology/data/cue_consistency.py`'s module docstring
    ("DELIBERATE DEPARTURES") for the twin generator that caught this. No published cells exist yet,
    so this is a pre-campaign fix: data for a given seed differs from the old code's output (the draw
    sequence changed), and that is intentional here and not to be repeated once cells are published.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    rng = np.random.default_rng(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2 * num_passes

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_idx = _distinct_per_row(rng, num_examples, len(key_choices), num_kv_pairs)
    keys = key_choices[keys_idx]

    values_idx = _distinct_per_row(rng, num_examples, len(value_choices), num_kv_pairs)
    values = value_choices[values_idx]

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    kvs = np.tile(kvs, (1, num_passes))

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    gaps = _weighted_distinct(rng, num_examples, p, num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs,
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs_np, labels_np = examples[:, :-1], labels[:, 1:]

    # replace all the 0 with random values, drawn from THIS generator's stream (never torch's
    # global one — that was the bug: `torch.randint` here read whatever else had touched torch's
    # global RNG first in the process, so the filler was not reproducible from `seed` alone).
    if random_non_queries:
        holes = inputs_np == 0
        inputs_np = inputs_np.copy()
        inputs_np[holes] = rng.integers(0, vocab_size, size=int(holes.sum()))

    inputs, labels = torch.tensor(inputs_np), torch.tensor(labels_np)
    return DataSegment(
        inputs, 
        labels, 
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len, "num_passes": num_passes}
    )

