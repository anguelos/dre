import pytest
import sys
import numpy as np 
sys.path.append("./")
from dre import FAGraph, NFASegment, WordParser
from pathlib import Path

regex_pdffilesizes = [("hello great? world", 1000)]

@pytest.mark.parametrize(("regex_text", "pdf_file_size"), regex_pdffilesizes)
def test_plot(regex_text, pdf_file_size):
    graph = WordParser().parse_str(regex_text)
    gv_file = "/tmp/test_fagraph.gv"
    graph.plot(filename=gv_file, view=False)
    Path(f"{gv_file}.pdf").stat().st_size == pdf_file_size
    print(graph)