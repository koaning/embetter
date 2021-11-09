<img src="icon.png" width="125" height="125" align="right" />

# embetter

> "The learning is in the labels."

<br> 

Embetter implements light neural networks that makes it easier for a human to be in the loop. The
human shouldn't conform to a labelling scheme, the learning system needs to learn fro the easiest
to produce labels. These labels typically include something that resembles a yes/no statement.

## warning 

I like to build in public, but please don't expect anything yet. This is alpha stuff!

## notes 

The objects to implement:

```python
Embetter(epochs=50)
  .fit_sim(X1, X2, y_sim)
  .partial_fit_sim(X1, X2, y_sim)
  .predict_sim(X1, X2)
  .embed(X)

Embsorter(epochs=50)
  .fit_order(X1, X2, y_geq)
  .partial_fit_order(X1, X2, y_geq)
  .predict_sim(X1, X2)
  .embed(X)
```
