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
