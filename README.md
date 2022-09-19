<img src="https://raw.githubusercontent.com/koaning/embetter/main/docs/images/icon.png" width="125" height="125" align="right" />

# embetter

> "Just a bunch of embeddings to get started quickly."

<br> 

Embetter implements scikit-learn compatible embeddings that should help get you started quickly.

## Install 

You can only install from Github, for now.

```
python -m pip install embetter
```

Many of the embeddings are optional depending on your use-case, so if you
want to nit-pick to download only the tools that you need: 

```
python -m pip install "embetter[text]"
python -m pip install "embetter[sense2vec]"
python -m pip install "embetter[sentence-tfm]"
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
from embetter.text import SentenceEncoder, Sense2VecEncoder
```

All of these components are scikit-learn compatible, which means that you
can apply them as you would normally in a scikit-learn pipeline. Just be aware
that these components are stateless. They won't require training as these 
are all pretrained tools. 

## Text Example

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
from embetter.vision import ImageLoader, TimmEncoder

# This pipeline grabs the `img_path` column from a dataframe
# then it grabs the image paths and turns them into `PIL.Image` objects
# which then get fed into MobileNetv2 via TorchImageModels (timm).
image_emb_pipeline = make_pipeline(
  ColumnGrabber("img_path"),
  ImageLoader(convert="RGB"),
  TimmEncoder("mobilenetv2_120d")
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

## Available Components 

The goal of the library is remain small but to offer a few general tools
that might help with bulk labelling in particular, but general scikit-learn
pipelines as well.

|       class               | link                                                 | What it does                                                                                          |
|:-------------------------:|------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `ColumnGrabber`           | [docs](https://koaning.github.io/embetter/API/grab/) | ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/columngrabber.png)            |
| `SentenceEncoder`         | [docs](https://koaning.github.io/embetter/API/text/sentence-enc/) | ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/sentence-encoder.png)         |
| `Sense2VecEncoder`        | [docs](https://koaning.github.io/embetter/API/text/sense2vec/) | ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/sense2vec.png)                |
| `ImageLoader`             | [docs](https://koaning.github.io/embetter/API/vision/imageload/) | ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/imageloader.png)              |
| `ColorHistogramEncoder`   | [docs](https://koaning.github.io/embetter/API/vision/colorhist/) | ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/colorhistogram.png)           |
| `TimmEncoder`             | [docs](https://koaning.github.io/embetter/API/vision/timm/) | ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/timm.png)                     |
