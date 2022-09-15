<img src="icon.png" width="125" height="125" align="right" />

# embetter

> "Just a bunch of embeddings to get started quickly."

<br> 

Embetter implements scikit-learn compatible embeddings that should help get you started quickly.

## Install 

For now you can only install from Github. 

```
python -m pip install "embetter @ git+https://github.com/koaning/embetter.git"
```

## API Design 

This is what's being implemented now. 

```python
# Helpers to grab text or image from pandas column.
from embetter.grab import ColumnGrabber
# Representations for computer vision
from embetter.image import ImageGrabber, Timm, ColorHistogram, Clip
# Representations for text
from embetter.text import SentenceTFM, Clip
```


## Text Example

```python
from sklearn.pipeline import make_pipeline 
from sklearn.linear import LogisticRegression

# This pipeline grabs the `text` column from a dataframe
# which then get fed into Sentence-Transformers' all-MiniLM-L6-v2.
text_emb_pipeline = make_pipeline(
  ListGrabber("text"),
  SBERT('all-MiniLM-L6-v2')
)

# This pipeline can also be trained to make predictions, using
# the embedded features. 
text_clf_pipeline = make_pipeline(
  text_emb_pipeline,
  LogisticRegression()
)

text_emb_pipeline.fit_transform(dataf, dataf['label_col'])
text_clf_pipeline.fit_predict(dataf, dataf['label_col'])
```

## Image Example

The goal of the API is to allow pipelines like this: 

```python
from sklearn.pipeline import make_pipeline 
from sklearn.linear import LogisticRegression

# This pipeline grabs the `img_path` column from a dataframe
# then it grabs the image paths and turns them into `PIL.Image` objects
# which then get fed into MobileNetv2 via TorchVision.
image_emb_pipeline = make_pipeline(
  ListGrabber("img_path"),
  ImageGrabber(convert="RGB"),
  TorchVision("pytorch/vision", "mobilenet_v2", "MobileNet_V2_Weights.IMAGENET1K_V2")
)

# This pipeline can also be trained to make predictions, using
# the embedded features. 
image_clf_pipeline = make_pipeline(
  image_emb_pipeline,
  LogisticRegression()
)

image_emb_pipeline.fit_transform(dataf, dataf['label_col'])
image_clf_pipeline.fit_predict(dataf, dataf['label_col'])
```
