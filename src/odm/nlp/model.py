from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import (DocumentPoolEmbeddings,
                              FlairEmbeddings, StackedEmbeddings)

from torch import dot, norm, squeeze

from pathlib import Path
import pickle


from odm.nlp.tensors import PCA, plot_embeddings, normalize_tensors
from odm.nlp.parse import parse_veclang


class Model:

    modelpath = '/data/model/'

    def __init__(self):

        # Sequence Tagging Model
        tagger_file = self.modelpath + 'tagger.pt'
        if Path(tagger_file).is_file():
            print('loading tagger from file')
            self.tagger = SequenceTagger.load_from_file(tagger_file)
        else:
            print('downloading pretrained tagger')
            self.tagger = SequenceTagger.load('ner-ontonotes')
            self.tagger.save(tagger_file)

        # Text Embedding Model
        embeddings_file = self.modelpath + 'embeddings.pickle'
        if Path(embeddings_file).is_file():
            print('loading embedder from file')
            filestream = open(embeddings_file, 'rb')
            self.embeddings = pickle.load(filestream)
        else:
            print('downloading pretrained embedders')
            self.embeddings = [
                # WordEmbeddings('glove'),
                FlairEmbeddings('multi-forward')
                # FlairEmbeddings('multi-backward')
            ]
            filestream = open(embeddings_file, 'wb')
            pickle.dump(self.embeddings, filestream)

        self.token_embedder = StackedEmbeddings(self.embeddings)
        self.doc_embedder = DocumentPoolEmbeddings(self.embeddings)

    def parse(self, text):

        sentence = Sentence(text)
        self.tagger.predict(sentence)
        self.token_embedder.embed(sentence)
        self.doc_embedder.embed(sentence)

        return sentence

    def mindmap(self, text):

        # parsing
        lines, arrows = parse_veclang(text)
        sentences = [self.parse(line) for line in lines]
        tensors = [s.get_embedding() for s in sentences]

        # tensor processing
        norm_tensors = normalize_tensors(tensors)
        flat_tensors = PCA(norm_tensors)

        # plot map
        filename = plot_embeddings(lines, flat_tensors, arrows)

        return f'Plotted mindmap to {filename}'

    def similarity(self, text):
        lines = text.split('//')
        sentences = [self.parse(line) for line in lines]
        vecs = [squeeze(s.embedding) for s in sentences]
        sim = dot(vecs[0], vecs[1])/(norm(vecs[0])*norm(vecs[1]))

        return f'the similarity is {sim}'
