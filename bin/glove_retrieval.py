import re
import sys
import numpy as np
import time


def load_embedding(csv_path):
    lines=[l.split(" ") for l in open(csv_path).read().strip().split("\n")]
    return {l[0]:np.array([float(col) for col in l[1:]]) for l in lines}

def load_coprpus(json_path):
    return [entry["caption"] for entry in json.load(open(json_path))["annotations"]]

def embed_document(txt,embed_dict):
    word_re=re.compile(["[A-Za-z]+"])
    unknown_embeding=np.ones(50)
    lines=[[word.lower() for word in word_re.findall(line)] for line in txt]
    embedded_lines=[[embed_dict.get(word,unknown_embeding) for word in line] for line in lines]
    return embedded_lines


if __name__=="__main__":
    t=time.time()
    emb=load_embedding("./data/glove.6B.50d.txt")
    print "Loaded embedings: %d msec"%(int(1000*(time.time()-t)))
    corpus=open(sys.argv[1]).read()
    #voc=set([word.lower() for word in re.findall("[A-Za-z]+",corpus)])
    #missing=voc-set(emb.keys())
    print "Embedded %d words in %d msec" % (len(corpus.split()),int(1000 * (time.time() - t)))
