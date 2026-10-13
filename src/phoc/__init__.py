# The PHOC module is built step by step; each stepNN_*.py builds on the previous one and the package exports the latest.
#   step01_interface — PHOC, the abstract design API: parameters, inputs, column layout, weighting, distance
#   step02_overlap   — the computation: exact fractional overlap of characters and regions, counted or thresholded
#   step03_semantic  — PHOCToSemantic: a trainable map from PHOC to a semantic embedding space (train_semantic.py)
#   step04_decoder   — PHOCToString: a character-level decoder spelling the word a PHOC encodes (train_decoder.py)
#   step05_glove2phoc — GloveToPHOC: a trainable map from a semantic embedding back to a PHOC (examples/phoc)
from .step02_overlap import PHOC
from .step03_semantic import PHOCToSemantic
from .step04_decoder import PHOCToString
from .step05_glove2phoc import GloveToPHOC
