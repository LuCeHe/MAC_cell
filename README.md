# keras MAC cell

Keras adaptation of the article [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/pdf/1803.03067.pdf). 

### How to use it

The .zip *CLEVR_v1.0.zip* can be downloaded from [here](https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip) (18 Gb) or from the [official website](https://cs.stanford.edu/people/jcjohns/clevr/), and has to be saved in the data folder for the data generator to find everything automatically.

To run the full experiment, run the main.py file. To pick the MAC cell for another task, probably the easiest is to initialize the class completeMACmodel_simple, and retrieve the model as:

``` python
MAC = completeMACmodel_simple()
model = MAC.model
```

to compile and fit a generator or a dataset after.

### DONEs and TODOs

Still work in progress, but maybe making it public can help me figure out what is missing.

List of DONEs:
- numpy arrays dimensions pass correctly before fitting
- gradients pass correctly
- training works, but the performance is still low

List of main TODOs:
- ResNet50 to ResNet101
- implement the full WriteUnit
- otherwise figure out what's the source of the low performance
