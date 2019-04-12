# Vectors of Locally Aggregated Concepts (VLAC)

## Method
As illustrated in the Figure below, VLAC clusters word embeddings to create *k* concepts. Due to the high dimensionality of word embeddings (i.e., 300) spherical k-means is used to perform the clustering as applying euclidean distance will result in little difference in the distances between samples. The method works as follows. Let *w_{i}* be a word embedding of size *D* assigned to cluster center c_{k}. Then, for each word in a document, VLAC computes the element-wise sum of residuals of each word embedding to its assigned cluster center. This results in *k* feature vectors, one for each concept, and all of size *D*. All feature vectors are then concatenated, power normalized, and finally, l2 normalization is applied. For example, if 10 concepts were to be created out of word embeddings of size 300 then the resulting document vector would contain 10 x 300 values. 

<img src="https://github.com/MaartenGr/VLAC/blob/master/vlac.png" width="70%"/>

