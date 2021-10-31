# embetter

> Improving Representations via Similarities

The object to implement:

```python
Embetter(multi_output=True, epochs=50, sampling_kwargs)
  .fit(X, y)
  .fit_sim(X1, X2, y_sim, weights)
  .partial_fit(X, y, classes, weights)
  .partial_fit_sim(X1, X2, y_sim, weights)
  .predict(X)
  .predict_proba(X)
  .predict_sim(X1, X2)
  .transform(X)
  .translate_X_y(X, y, classes=none)
```

**Observation**: *especially* when `multi_output=True` there's an opportunity with regards to `NaN` `y`-values. We can simply choose with values to translate and which to ignore. 
