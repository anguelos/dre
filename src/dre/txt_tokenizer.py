import re
import numpy as np
import torch

from .tokenizing import words_of_sentences


def clean_gutenberg_txt(txt_str:str):
    #chopping gutenberg
    r_start = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*")
    r_end = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*")
    txt_start = list(r_start.finditer(txt_str))[0].span()[1]+1
    txt_end = list(r_end.finditer(txt_str))[0].span()[0]
    txt_str = txt_str[txt_start:txt_end]
    txt_str = txt_str.replace("\n\n\n","@@").replace("\n\n","@@").replace("\n"," ").replace("@@","\n")
    return txt_str


class Glove():
    avg_50b = torch.tensor([ 0.072617, -0.51393 ,  0.4728  , -0.52202 , -0.35534 ,  0.34629 ,
        0.23211 ,  0.23096 ,  0.26694 ,  0.41028 ,  0.28031 ,  0.14107 ,
       -0.30212 , -0.21095 , -0.10875 , -0.33659 , -0.46313 , -0.40999 ,
        0.32764 ,  0.47401 , -0.43449 ,  0.19959 , -0.55808 , -0.34077 ,
        0.078477,  0.62823 ,  0.17161 , -0.34454 , -0.2066  ,  0.1323  ,
       -1.8076  , -0.38851 ,  0.37654 , -0.50422 , -0.012446,  0.046182,
        0.70028 , -0.010573, -0.83629 , -0.24698 ,  0.6888  , -0.17986 ,
       -0.066569, -0.48044 , -0.55946 , -0.27594 ,  0.056072, -0.18907 ,
       -0.59021 ,  0.55559 ])
    @staticmethod
    def load_glove(path="tmp/glove/glove.6B.50d.txt"):
        lines = [l.split() for l in open(path, "r").read().strip().split("\n")]
        embbeding_list = [(l[0], np.array([float(c) for c in l[1:]])) for l in lines]
        del lines
        embbeding_dict = {}
        for word, embeeding in embbeding_list:
            embbeding_dict[word] = torch.tensor(embeeding)
        return embbeding_dict, embbeding_list
    
    def embedding_length(self):
        return self.avg.shape[0]

    def __init__(self, glove_path="tmp/glove/glove.6B.50d.txt", tokenizer="auto") -> None:
        self.tokenizer = tokenizer
        self.embbeding_dict, self.embbeding_list = Glove.load_glove(glove_path)
        avg = np.zeros_like(self.embbeding_list[0][1])
        for _, embedding in self.embbeding_list:
            avg+=embedding
        self.avg = torch.tensor(avg / len(self.embbeding_list))

    def __getitem__(self, token):
        return self.embbeding_dict.get(token,self.avg)

    def embed_sentence(self, sentence):
        tokens = [token for words in words_of_sentences(sentence, tokenizer=self.tokenizer) for token in words]
        tokens = torch.cat([self[t][:,None] for t in tokens], dim=1)
        return tokens

    def get_observation_probabillities(self, sentence, row_wise_embeddings):
        sentence = self.embed_sentence(sentence)
        return torch.softmax(torch.matmul(row_wise_embeddings, sentence) *.5 + .5, dim=1)
        
