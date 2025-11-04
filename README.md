# SPoT: Subpixel Placement of Tokens in Vision Transformers

![SPoT Figure 1](/assets/placements.png#gh-light-mode-only "Examples of feature trajectoreis with SPoT-ON")
![SPoT Figure 1](/assets/placements.png#gh-dark-mode-only "Examples of feature trajectoreis with SPoT-ON")

This repo contains code and weights for **SPoT: Subpixel Placement of Tokens**

## Loading models

To load the model, first download the checkpoints from XXXX.
NB: Checkpoints will be released with the released non-anonymous repo.
Then extract the checkpoints into a folder named `checkpoints/` in the cloned repo.

The model can be loaded easily by

```
from spot.load_models import *

model_name = 'spot_mae_b16'
assert model_name in valid_models
model = load_trained_model(
    model_name=model_name,
    sampler='grid_center',      # Spatial prior
    ksize=16,                   # Window size
    n_features=25,              # Number of tokens
)
```


## More Examples

We provide a [Jupyter notebook](./get_started.ipynb) that illustrates loading, evaluating, and extracting token placements for the models. 