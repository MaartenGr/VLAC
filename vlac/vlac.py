import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
# from nltk.cluster import KMeansClusterer


class VLAC:
    """ Convert a collection of text documents to a matrix based on clustered word embeddings

    Vectors of Locally Aggregated Concepts (VLAC) is a method that leverages clusters of word
    embeddings (i.e., concepts) to create features from a collection of documents.

    k-Means is used to cluster word embeddings into k word clusters, which are named concepts
    by the similarity of words in a word cluster. For each document, the element-wise sum of
    residuals of each word embedding to its assigned cluster center is calculated.

    This results in k features vectors, one for each concepts, and all the same size as the
    dimensions of the word embeddings. All feature vectors are then concatenated, power normalized,
    and finally, l2 normalization is applied. For example, if 10 concepts were to be created out of
    word embeddings of size 300 then the resulting document vector would contain 10 x 300 values.
    """

    def __init__(self, documents, model, oov=False):

        self.documents = documents
        self.oov = oov
        self.model = model

        if type(model) == dict:
            self.model_word_list = list(model.keys())
            self.dim = len(model[list(model.keys())[0]])
        else:
            self.model_word_list = list(model.wv.vocab)
            self.dim = model.vector_size

    def fit_transform(self, num_concepts):
        """ Creates concepts (i.e., word clusters) and based on those concepts
        create a feature space from the concatenated sum of residuals in each
        concept.

        Parameters
        ----------
        num_concepts : int
            Number of concepts (i.e., word clusters) to be created using a
            spherical kmeans clustering task

        Returns
        -------
        vlac : numpy array
            A numpy matrix with dimensions (k x vector dimensions) x (number of documents)

        kmeans : sklearn.cluster.k_means_.KMeans
            A fitted sklearn kmeans model
        """
        word_vectors = np.array([self.model[word] for word in self.model_word_list])

        word_to_concept, concept_centers, kmeans = self._create_clusters(word_vectors,
                                                                         self.model_word_list,
                                                                         num_concepts)

        residuals = {word: word_vectors[i] - concept_centers[word_to_concept[word]]
                     for i, word in enumerate(self.model_word_list)}

        vlac = self._create_vlac_features(word_to_concept, residuals, num_concepts)
        return vlac, kmeans

    def transform(self, kmeans):
        """ From a fitted kmeans transform a collection of documents to VLAC features

        Parameters
        ----------
        kmeans : sklearn.cluster.k_means_.KMeans
            A fitted sklearn kmeans model

        Returns
        -------
        vlac : numpy array
            A numpy matrix with dimensions (k x vector dimensions) x (number of documents)
        """
        # Get all words in the document that also appear in the model
        # It does not check it if FastText is used which can handle out-of-vocabulary words
        if self.oov:
            word_list = list(set([word for doc in self.documents for word in doc.split()]))
        else:
            word_list = list(set([word for doc in self.documents for word in doc.split()
                                  if word in self.model_word_list]))

        # create word_vectors for each word in the word list
        word_vectors = [self.model[word] for word in word_list]

        # then check which concept they belong to based on the concept_centers
        predicted_concepts = kmeans.predict(preprocessing.normalize(word_vectors))
        word_to_concept = {word: concept for word, concept in zip(word_list,
                                                                  predicted_concepts)}
        concept_centers = kmeans.cluster_centers_

        # Calculate residuals
        residuals = {word: self.model[word] - concept_centers[word_to_concept[word]]
                     for word in word_list}

        vlac = self._create_vlac_features(word_to_concept, residuals, kmeans.n_clusters)

        return vlac

    def _create_clusters(self, word_vectors, word_list, num_concepts):
        """" It creates clusters on the basis of words in the model
        It is up to you if you want to get a pre-trained model and cluster
        on all those words or take the pre-trained model and only select
        words that appear in your documents.

        Parameters
        ----------
        word_vectors : numpy array
            Array of word embeddings of all words in the word_list
            Indices between word_vectors and word_list match for easier lookup

        word_list : array
            Array of strings containing the names of words in the word_vectors

        num_concepts : int
            Number of word clusters (i.e., concepts) to be created

        """
        # normalizing results in spherical kmeans
        normalized_vectors = preprocessing.normalize(word_vectors)
        kmeans = KMeans(n_clusters=num_concepts)
        kmeans.fit(normalized_vectors)

        concept_centers = kmeans.cluster_centers_
        word_to_concept = {word: concept for word, concept in zip(word_list, kmeans.labels_)}

        return word_to_concept, concept_centers, kmeans

    def _create_vlac_features(self, word_to_concept, residuals, num_concepts):
        """ Transform documents to VLAc features

        For each document, the element-wise sum of residuals of each word embedding to its
        assigned cluster center is calculated.  This results in k features vectors,
        one for each concepts, and all the same size as the dimensions of the word embeddings.
        All feature vectors are then concatenated, power normalized, and finally, l2 normalization
        is applied. For example, if 10 concepts were to be created out of
        word embeddings of size 300 then the resulting document vector would contain 10 x 300 values.

        Parameters
        ----------
        word_to_concept : dict
            Dictionary containing word (key) and concept (value) for quick lookup

        residuals : dict
            Dictionary containing word (key) and word embedding (value) for quick lookup

        num_concepts : int
            Number of word clusters (i.e., concepts) that were created

        Returns
        -------
        vlac : numpy array
            Matrix containing vlac features for each document
        """
        vlac = []

        for line in self.documents:
            document_vector = np.zeros([num_concepts, self.dim])
            for word in line.split():
                try:
                    document_vector[word_to_concept[word]] += residuals[word]
                except KeyError:
                    continue

            # concatenate vectors
            document_vector = document_vector.flatten()

            # power normalization, also called square-rooting normalization
            document_vector = np.sign(document_vector) * np.sqrt(np.abs(document_vector))

            # L2 normalization
            l2_norm = np.sqrt(np.dot(document_vector, document_vector))
            if l2_norm != 0:
                document_vector = np.divide(document_vector, l2_norm)

            vlac.append(document_vector)

        return np.array(vlac)
