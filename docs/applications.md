---
title: Applications
---

This document contains some tricks, hints and demos of applications that you might want to consider
in combination with this library. 

## Speedup with Modal 

Embedding text can be slow, especially when you're running on a CPU. If you wish 
to speed up your embedding calculations you may enjoy using [modal](https://modal.com/). 
Modal allows you to add a GPU to a Python function simply by adding a decorator.

Not every encoder in embetter will get a speedup by using a GPU but the
`SentenceEncoder` as well as `ClipEncoder` should both automatically detect
when the GPU is available automatically.

The code below gives an example. 

```python
import time
import h5py
import modal


stub = modal.Stub("example-get-started")
image = (modal.Image.debian_slim()
         .pip_install("simsity", "embetter[text]", "h5py")
         .run_commands("python -c 'from embetter.text import SentenceEncoder; SentenceEncoder()'"))


# This is the function that actually runs the embedding, 
# notice that there's a GPU attached.
@stub.function(image=image, gpu="any")
def create(data):
    from embetter.text import SentenceEncoder
    return SentenceEncoder().transform(data)


@stub.local_entrypoint()
def main():
    tic = time.time()

    # You'd need to write your own function to read in the texts
    data = read_text()
    
    # This runs our decorated function on external hardware
    X = create.call(data)

    # Next we save it to disk for re-use
    with h5py.File('embeddings.h5', 'w') as hf:
        hf.create_dataset("embeddings",  data=X)
    toc = time.time()
    print(f"took {toc - tic}s to embed shape {X.shape}")
```

On our own benchmarks, we seem to get a 4-5x speedup with just a minor edit
to the code. This can be extremely helpful when you're trying to embed data
in bulk.
