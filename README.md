<img src="icon.png" width="125" height="125" align="right" />

# embetter

> "Just a bunch of embeddings to get started quickly."

<br> 

Embetter implements scikit-learn compatible embeddings that should help get you started quickly.

## API Design 

This is what's being implemented now. 

```python
# Helpers to grab text or image from pandas column.
from embetter.grab import ListGrabber, ImageGrabber
# Representations for computer vision
from embetter.image import TorchVision
# Representations for text
from embetter.text import SBERT
```


## Text Example

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
```
