# Vectors of Locally Aggregated Concepts (VLAC)

[![PyPI - Status](https://img.shields.io/badge/status-beta-yellow.svg)]()
[![PyPI - Python](https://img.shields.io/badge/python-3.4%20%7C%203.5%20%7C%203.6-blue.svg)]()
[![PyPI - Python](https://img.shields.io/badge/license-MIT-green.svg)]()


## Installation

Installation can be done using [pypi](https://pypi.org/project/vlac/)

``pip install pm-bum``

## Purpose
As illustrated in the Figure below, VLAC clusters word embeddings to create *k* concepts. Due to the high dimensionality of word embeddings (i.e., 300) spherical k-means is used to perform the clustering as applying euclidean distance will result in little difference in the distances between samples. The method works as follows. Let *w<sub>i</sub>* be a word embedding of size *D* assigned to cluster center *c<sub>k</sub>*. Then, for each word in a document, VLAC computes the element-wise sum of residuals of each word embedding to its assigned cluster center. This results in *k* feature vectors, one for each concept, and all of size *D*. All feature vectors are then concatenated, power normalized, and finally, l2 normalization is applied. For example, if 10 concepts were to be created out of word embeddings of size 300 then the resulting document vector would contain 10 x 300 values. 

<img src="https://github.com/MaartenGr/VLAC/blob/master/Images/vlac.png?raw=true" width="70%"/>

## Usage
Tested in python 3.5.4. 

```python
# Train model and transform collection of documents
vlac_model = VLAC(documents=train_docs, model=model, oov=False)
vlac_features, kmeans = vlac_model.fit_transform(num_concepts=30)

# Create features new documents
vlac_model = VLAC(documents=train_docs, model=model, oov=False)
test_features = vlac_model.transform(kmeans=kmeans)
```

