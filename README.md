
# embetter

> "Just a bunch of useful embeddings for scikit-learn pipelines, to get started quickly."

<img src="https://raw.githubusercontent.com/koaning/embetter/main/docs/images/icon.png" width="125" height="125" align="right" />

<br> 

Embetter implements scikit-learn compatible embeddings for computer vision and text. It should make it very easy to quickly build proof of concepts using scikit-learn pipelines and, in particular, should help with [bulk labelling](https://www.youtube.com/watch?v=gDk7_f3ovIk). It's also meant to play nice with [bulk](https://github.com/koaning/bulk) and [scikit-partial](https://github.com/koaning/scikit-partial) but it can also be used together with your favorite ANN solution like [lancedb](https://lancedb.github.io/lancedb/).

## Install 

You can install via pip.

```
python -m pip install embetter
```

Many of the embeddings are optional depending on your use-case, so if you
want to nit-pick to download only the tools that you need: 

```
python -m pip install "embetter[text]"
python -m pip install "embetter[spacy]"
python -m pip install "embetter[sense2vec]"
python -m pip install "embetter[vision]"
python -m pip install "embetter[all]"
```

## API Design 

This is what's being implemented now. 

```python
# Helpers to grab text or image from pandas column.
from embetter.grab import ColumnGrabber

# Representations/Helpers for computer vision
from embetter.vision import ImageLoader, TimmEncoder, ColorHistogramEncoder

# Representations for text
from embetter.text import SentenceEncoder, MatryoshkaEncoder, Sense2VecEncoder, spaCyEncoder, TextEncoder

# Representations from multi-modal models
from embetter.multi import ClipEncoder

# Finetuning components 
from embetter.finetune import FeedForwardTuner, ContrastiveTuner, ContrastiveLearner, SbertLearner

# External embedding providers, typically needs an API key
from embetter.external import CohereEncoder, OpenAIEncoder
```

All of these components are scikit-learn compatible, which means that you
can apply them as you would normally in a scikit-learn pipeline. Just be aware
that these components are stateless. They won't require training as these 
are all pretrained tools. 

## Text Example

To run this example, make sure that you `pip install 'embetter[sbert]'`. 

```python
import pandas as pd
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LogisticRegression

from embetter.grab import ColumnGrabber
from embetter.text import SentenceEncoder

# This pipeline grabs the `text` column from a dataframe
# which then get fed into Sentence-Transformers' all-MiniLM-L6-v2.
text_emb_pipeline = make_pipeline(
  ColumnGrabber("text"),
  SentenceEncoder('all-MiniLM-L6-v2')
)

# This pipeline can also be trained to make predictions, using
# the embedded features. 
text_clf_pipeline = make_pipeline(
  text_emb_pipeline,
  LogisticRegression()
)

dataf = pd.DataFrame({
  "text": ["positive sentiment", "super negative"],
  "label_col": ["pos", "neg"]
})
X = text_emb_pipeline.fit_transform(dataf, dataf['label_col'])
text_clf_pipeline.fit(dataf, dataf['label_col']).predict(dataf)
```

## Image Example

The goal of the API is to allow pipelines like this: 

```python
import pandas as pd
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LogisticRegression

from embetter.grab import ColumnGrabber
from embetter.vision import ImageLoader
from embetter.multi import ClipEncoder

# This pipeline grabs the `img_path` column from a dataframe
# then it grabs the image paths and turns them into `PIL.Image` objects
# which then get fed into CLIP which can also handle images.
image_emb_pipeline = make_pipeline(
  ColumnGrabber("img_path"),
  ImageLoader(convert="RGB"),
  ClipEncoder()
)

dataf = pd.DataFrame({
  "img_path": ["tests/data/thiscatdoesnotexist.jpeg"]
})
image_emb_pipeline.fit_transform(dataf)
```

## Batched Learning 

All of the encoding tools you've seen here are also compatible
with the [`partial_fit` mechanic](https://scikit-learn.org/0.15/modules/scaling_strategies.html#incremental-learning) 
in scikit-learn. That means
you can leverage [scikit-partial](https://github.com/koaning/scikit-partial)
to build pipelines that can handle out-of-core datasets. 

