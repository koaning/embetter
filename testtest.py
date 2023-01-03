from datasets import load_dataset
imdb = load_dataset("imdb")

from embetter.text import SentenceEncoder

# Load up a sentence encoder.
enc = SentenceEncoder()

# Assume we have 200 labels 
n_train = 200

# Grab 200 examples and encode them
df_train = imdb['train'].to_pandas().sample(frac=1, random_state=32)
X_train = enc.transform(df_train['text'].to_list()[:n_train])
y_train = df_train['label'][:n_train].values

# Let's grab 2000 examples for our "test" set 
n_test = 2000

# Grab 2000 examples and encode them
df_test = imdb['test'].to_pandas().sample(frac=1, random_state=42)
X_test = enc.transform(df_test['text'].to_list()[:n_test])
y_test = df_test['label'][:n_test].values

from embetter.finetune import ForwardFinetuner 

# Create a network with some settings. You can totally change these. 
tuner = ForwardFinetuner(n_epochs=500, learning_rate=0.01, hidden_dim=200)

# Learn from our small training data
tuner.fit(X_train, y_train)

# Note that it's all skearn compatible 
X_test_tfm = tuner.transform(X_test)

from sklearn.decomposition import PCA
from matplotlib import pylab as plt 

X_orig = PCA().fit_transform(X_test)
X_finetuned = PCA().fit_transform(X_test_tfm)

# First chart 
plt.scatter(X_orig[:, 0] , X_orig[:, 1], c=y_test, s=10)
plt.title("PCA of original embedding space")

# Second chart
plt.scatter(X_finetuned[:, 0] , X_finetuned[:, 1], c=y_test, s=10)
plt.title("PCA of fine-tuned embedding space")

tuner = ForwardFinetuner(n_epochs=500, learning_rate=0.01, hidden_dim=10)

from sklearn.pipeline import make_pipeline 

# Grab a few examples
X = df_test['text'].to_list()[:50]
y = df_test['label'].to_list()[:50]

# Let's build a pipeline!
pipe = make_pipeline(
    SentenceEncoder(),
    ForwardFinetuner(n_epochs=500, learning_rate=0.01, hidden_dim=10),
    PCA()
)

# The fine-tuning component can use `y_train`.
pipe.fit(X, y)

# Apply all the trained steps! 
pipe.transform(X)