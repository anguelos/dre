import sys
sys.path.append("./dre")
from abstract_nfa import FAGraph

from sly import Lexer
from sly import Parser
from abstract_nfa import FAGraph
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import Union



class WordLexer(Lexer):
    # This is the set of tokens we are exporting to the Parser
    tokens = {WORD, ASTERISK, QUESTIONMARK, PLUS, OR, LEFT_PAR, RIGHT_PAR}
    # Any literals we want to ignore
    ignore = ' \t'
    # Any literals we did not define as tokens, will be available for usage in the Parser
    literals = {'.', '!'}

    # The definition of each token in a regex pattern - Notice that the order MATTERS!! First match will be taken
    WORD = r'[A-Za-z]+'
    ASTERISK = r'\*'
    QUESTIONMARK = r'\?'
    PLUS = r'\+'
    OR = r'\|'
    LEFT_PAR = r'\('
    RIGHT_PAR = r'\)'


class WordParser(Parser):
    tokens = WordLexer.tokens

    # Un-comment the following line to output the parser logs for debugging any conflicts
    debugfile = 'parser.out'

    # This sets the order of execution. The last values will have a higher precedence
    precedence = (
        ('left', ASTERISK, QUESTIONMARK, PLUS),
        ('left', OR),
        ('left', RIGHT_PAR),
        ('right', LEFT_PAR),
    )

    def __init__(self, variables: dict = None):
        self.variables = variables or {}  # We want to allow input context
        self.stack = []  # We will use a stack to traverse the tree and keep the results

    @property
    def last_item_on_stack(self):
        return self.stack[-1] if len(self.stack) > 0 else None

    @_("orexpr","literal", "iexpr", "uexpr")
    def expr(self, p):
        #print ("EXPR", tuple(p))
        #return FAGraph.NFASegment.create_transition_segment(symbol_str=p[0], caption=p[0])
        if isinstance(p[0], FAGraph.NFASegment):
            return p[0]
        elif isinstance(p[0], str):
            return FAGraph.NFASegment.create_transition_segment(symbol_str=p[0], caption=f"expr:{p[0]}")
        else:
            raise ValueError


    
    @_("expr expr")
    def expr(self, p):
        print("E_E", tuple(p))
        return FAGraph.NFASegment.create_concatenate_segment(p[0], p[1])
        

    @_("expr OR expr")
    def orexpr(self, p):
        print("OR", tuple(p))
        return FAGraph.NFASegment.create_or_segment(p[0], p[2])

    @_("expr ASTERISK")
    def uexpr(self, p):
        print("ASTERISK", tuple(p))
        return FAGraph.NFASegment.create_asterisk_segment(p[0])

    @_("expr QUESTIONMARK")
    def uexpr(self, p):
        print ("QUESTIONMARK", tuple(p))
        return FAGraph.NFASegment.create_question_segment(p[0])

    @_("expr PLUS")
    def uexpr(self,p):
        print("PLUS", tuple(p))
        return FAGraph.NFASegment.create_plus_segment(p[0])

    @_("literal")
    def iexpr(self, p):
        print ("LITERAL", tuple(p))
        return FAGraph.NFASegment.create_transition_segment(symbol_str=p[0], caption=p[0])

    @_("LEFT_PAR expr RIGHT_PAR")
    def iexpr(self,p):
        print(f"Parenthesis:{repr(p)}")
        return FAGraph.NFASegment.create_transition_segment(symbol_str="P[1]", caption="PARENTHESIS")

    @_('WORD')
    def literal(self,p):
        print(f"LITERAL:{repr(p[0])}")
        return(p[0])
        #self.stack.append(p[0])


def main():
    lexer = WordLexer()
    while True:
        text = input(">>> ")
        tokens = lexer.tokenize(text) # Creates a generator of tokens
        parser = WordParser()
        graph = FAGraph()
        graph.re_str=text
        graph.nfa_segment=parser.parse(tokens)
        graph.nfa_segment.label_states()
        #graph.draw_graph(graph.nfa_segment)
        print(graph)
        print(graph.create_matrices())
        #print(parser.parse(tokens)) # The entry point to the parser
        #print(parser.last_item_on_stack)

if __name__ == '__main__':
    main()
