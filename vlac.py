import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn import preprocessing

class VLAC():

    def __init__(self, doc_path, dim, num_concept, model):
        self.doc_path=doc_path
        self.dim=dim
        self.num_concept=num_concept
        self.model=model
        
        if type(self.model) == dict:
            self.wlist = list(self.model.keys())
        else:
            self.wlist = list(self.model.wv.vocab)
        
    def _get_wordvectors(self, model, wlist):
        '''
        Input: W2V model, list of words above minimum frequency
        Output: W2V Matrix
        '''
        w2v=list()
        for word in wlist:
            w2v.append(model[word])
        return np.array(w2v)

    def _create_concepts(self, w2vM, wlist):
        '''
        Input: W2V Matrix, word list above min freq, output path, # of concepts
        Ouput: File containing (word,concept)
        '''

        X_Norm = preprocessing.normalize(w2vM)
        skm=KMeans(n_clusters=self.num_concept)
        skm.fit(X_Norm)

        cluster_centers = skm.cluster_centers_
        word2concept={}
        for w,c in zip(wlist,skm.labels_):
            word2concept[w]=c
        print(".... Words have been assigned to concepts")
        return word2concept, cluster_centers
    

    def _create_vlac_features(self, word2concept, residuals):
        print('apply cfidf')
        boc_matrix=[]
        boc_matrix_count=[]
        count = 0
        start= time.time()
        with open(self.doc_path, "r") as f:
            for line in f:
                if count%1000 == 0:
                    end = time.time()
                    print(str(count) + '\t' + str(end-start))
                    start = time.time()
                    
                document_vector_count=np.zeros(self.num_concept)
                document_vector=np.zeros([self.num_concept,self.dim])
            
                for word in line.split():
                    try:    
                        document_vector[word2concept[word]] += residuals[word]
                
                    except KeyError:
                        continue

                document_vector = document_vector.flatten()
                
                # power normalization, also called square-rooting normalization
                document_vector = np.sign(document_vector)*np.sqrt(np.abs(document_vector))

                # L2 normalization
                document_vector = document_vector/np.sqrt(np.dot(document_vector,document_vector))

                boc_matrix.append(document_vector)
                count+=1
        print("Finished with applying VLAD")
        return np.array(boc_matrix)

    def _calc_residuals(self, model, cluster_centers, word2concept, wM, wlist):
        print('calculate distance')
        residuals = {word:wM[i]-cluster_centers[word2concept[word]] for i, word in enumerate(wlist)}
        return residuals
    
    def create_vlac_features(self):
        wM = self._get_wordvectors(self.model,self.wlist)
        word2concept, cluster_centers = self._create_concepts(wM,self.wlist) 
        residuals = self._calc_residuals(self.model, cluster_centers, word2concept, wM, self.wlist)
        vlac_features = self._create_vlac_features(word2concept,residuals)
        return vlac_features