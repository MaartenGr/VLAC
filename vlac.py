import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing


class VLAC:

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
        word_vectors = np.array([self.model[word] for word in self.model_word_list])

        word_to_concept, concept_centers, kmeans = self._create_clusters(word_vectors,
                                                                         self.model_word_list,
                                                                         num_concepts)

        residuals = {word: word_vectors[i] - concept_centers[word_to_concept[word]]
                     for i, word in enumerate(self.model_word_list)}

        vlac = self._create_vlac_features(word_to_concept, residuals, num_concepts)
        return vlac, kmeans

    def transform(self, kmeans):
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

        """
        # normalizing results in spherical kmeans
        normalized_vectors = preprocessing.normalize(word_vectors)

        kmeans = KMeans(n_clusters=num_concepts)
        kmeans.fit(normalized_vectors)

        concept_centers = kmeans.cluster_centers_
        word_to_concept = {word: concept for word, concept in zip(word_list, kmeans.labels_)}

        return word_to_concept, concept_centers, kmeans

    def _create_vlac_features(self, word_to_concept, residuals, num_concepts):
        boc_matrix = []

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

            boc_matrix.append(document_vector)

        return np.array(boc_matrix)
