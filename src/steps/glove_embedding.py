import numpy as np

class glove_embedding:
    def __init__(self):
        vocabs,embeddings = [],[]
        with open('../../utils/glove.6B.300d.txt','rt',encoding="utf8") as fi:
            full_content = fi.read().strip().split('\n')
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocabs.append(i_word)
            embeddings.append(i_embeddings)
        self.vocabs = vocabs
        self.embeddings = embeddings

    def get_embedding(self):
        # convert to array
        vocabs = np.array(self.vocabs)
        embeddings = np.array(self.embeddings)

        # insert <pad> and <unk> to vocab
        vocabs = np.insert(vocabs, 0, '<pad>')
        vocabs = np.insert(vocabs, 1, '<unk>')

        pad_emb = np.zeros((1,embeddings.shape[1]))   #embedding for '<pad>' token.
        unk_emb = np.mean(embeddings,axis=0,keepdims=True)    #embedding for '<unk>' token.

        #insert embeddings for pad and unk tokens at top of embeddings.
        embs_npa = np.vstack((pad_emb,unk_emb,embeddings))

        # Making a dictionary of vocabs and index related to them
        word_to_index = dict(zip(list(vocabs),range(len(vocabs))))

        return vocabs, embeddings, word_to_index