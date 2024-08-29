import numpy as np

class NaiveBayes:
    def __init__(self, binary_count=False):
        self.voc = set() # vocabulary in X_train
        self.binary = binary_count


    def fit(self, X_train, y_train):
        # identify unique classes and calculate prior probabilities p(c) for each class
        self.classes, counts = np.unique(y_train, return_counts=True)
        self.Pcs = [np.log(ci/len(X_train)) for ci in counts]
        #print(f'Pcs: {self.Pcs}')

        # calculate bag of words per class
        self.bags = []
        self.bags_len = []
        for i, ci in enumerate(self.classes):
            #print(f'class: {ci}')
            
            # documents of each class
            docs = X_train[y_train==ci]
            #print(docs)

            # concatenate docs and identify unique words and count
            concat = []
            for di in docs:
                concat += self.tokenize(di)
            
            self.bags_len.append(len(concat)) # bag len

            ks, vs = np.unique(concat, return_counts=True)
            auxd = {}
            for j in range(len(ks)):
                auxd[ks[j]] = vs[j]
            self.bags.append(auxd) # add bag of words
            #print(f'bag: {auxd}')
        #print(f'bags: {self.bags}\n len: {self.bags_len}')
    

    def count(self, w, ci):
        if w in self.bags[ci]: # if w is in ci bag of words
            return self.bags[ci][w]
        else: # if w not in the ci bag of words, return 0
            return 0


    def predict(self, X_test):
        preds = []

        for xi in X_test:
            auxp = []
            for ci, cl in enumerate(self.classes):
                #print(f'class: {cl}')
                # class probability
                priorc = self.Pcs[ci]

                # likelohood probability
                lp = 0
                for wi in self.tokenize(xi, unk=False):
                    lp = lp + np.log((self.count(wi, ci) + 1) / (self.bags_len[ci] + len(self.voc)))
                probc = priorc + lp # probability of xi belonginf to class c
                auxp.append(probc)
                #print(f'prob: {probc}')
            
            # choose max probability
            maxi = np.argmax(auxp)
            preds.append(self.classes[maxi])
        
        return preds


    def tokenize(self, text, unk=True):
        # eliminate symbols on input text
        sims = "!\"#$%&()*+-.,'/:;<=>?@[\]^_`{|}~\n"
        for si in sims:
            text = text.replace(si, '')

        # lower text
        text = text.lower()

        # separate text by words
        words = text.split(' ')

        # add words to vocabulary
        if unk:
            for wi in words:
                if wi not in self.voc:
                    self.voc.add(wi)
        else:
            aux = [] # only add words in vocabulary
            for wi in words:
                if wi in self.voc:
                    aux.append(wi)
            words = aux
        
        # binary version
        if self.binary:
            return list(np.unique(words))
        else:
            return words