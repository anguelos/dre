from .fa_segment import NFASegment
from .fa_graph import ANY as ANY_SYMBOL, Automaton, FAGraph

from sly import Lexer,  Parser

from matplotlib import pyplot as plt
from collections import defaultdict
from typing import Union



class WordLexer(Lexer):
    # This is the set of tokens we are exporting to the Parser
    tokens = {ANY, START, END, WORD, CLASS, ASTERISK, QUESTIONMARK, PLUS, OR, LEFT_PAR, RIGHT_PAR}
    # Any literals we want to ignore
    ignore = ' \t'
    # Any literals we did not define as tokens, will be available for usage in the Parser
    literals = {'.', '!'}

    # The definition of each token in a regex pattern - Notice that the order MATTERS!! First match will be taken
    ANY = r'\.'
    START = r'\^'
    END = r'\$'
    WORD = r"[A-Za-z'`]+"
    CLASS = r'<[A-Za-z_]+>'  # a token class such as <number>; which tokens belong to it is decided by the caller
    ASTERISK = r'\*'
    QUESTIONMARK = r'\?'
    PLUS = r'\+'
    OR = r'\|'
    LEFT_PAR = r'\('
    RIGHT_PAR = r'\)'

    def error(self, t):
        raise ValueError(f"Unexpected character {t.value[0]!r} at position {t.index} of a word regex")


class WordParser(Parser):
    tokens = WordLexer.tokens

    # Un-comment the following line to output the parser logs for debugging any conflicts
    # debugfile = 'parser.out'

    # This sets the order of execution. The last values will have a higher precedence:
    # alternation binds loosest, then concatenation (CONCAT, used via %prec), then the postfix operators. Tokens that
    # start an operand share the level of concatenation: without a level, sly prefers reducing "a | b" to shifting "c".
    precedence = (
        ('left', OR),
        ('left', CONCAT, WORD, CLASS, ANY),
        ('left', ASTERISK, QUESTIONMARK, PLUS),
        ('left', RIGHT_PAR),
        ('right', LEFT_PAR),
    )

    def __init__(self):
        super().__init__()
        self.lexer = WordLexer()
    
    def parse_str(self, text):
        tokens = self.lexer.tokenize(text) # Creates a generator of tokens
        result = self.parse(tokens)
        if result is None:
            raise ValueError(f"Empty word regex: {text!r}")
        result.re_str = text
        return result

    def error(self, p):
        raise ValueError(f"Unexpected {p.value!r} in a word regex" if p else "Unexpected end of a word regex")

    @property
    def last_item_on_stack(self):
        return self.stack[-1] if len(self.stack) > 0 else None

    @_('START expr END')
    def regex(self, p):
        segment = p[1]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=True, end=True)

    @_('START expr')
    def regex(self, p):
        segment = p[1]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=True, end=False)

    @_('expr END')
    def regex(self, p):
        segment = p[0]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=False, end=True)

    @_('expr')
    def regex(self, p):
        segment = p[0]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=False, end=False)

    @_("orexpr", "iexpr", "uexpr")  # literals arrive through iexpr; listing them here too was a reduce/reduce conflict
    def expr(self, p):
        #print ("EXPR", tuple(p))
        #return NFASegment.create_transition_segment(symbol_str=p[0], caption=p[0])
        if isinstance(p[0], NFASegment):
            return p[0]
        elif isinstance(p[0], str):
            return NFASegment.create_transition_segment(symbol_str=p[0], caption=f"S:{p[0]}")
        else:
            raise ValueError

    @_("expr expr %prec CONCAT")
    def expr(self, p):
        return NFASegment.create_concatenate_segment(p[0], p[1])
        

    @_("expr OR expr")
    def orexpr(self, p):
        return NFASegment.create_or_segment(p[0], p[2])

    @_("expr ASTERISK")
    def uexpr(self, p):
        return NFASegment.create_asterisk_segment(p[0])

    @_("expr QUESTIONMARK")
    def uexpr(self, p):
        return NFASegment.create_question_segment(p[0])

    @_("expr PLUS")
    def uexpr(self,p):
        return NFASegment.create_plus_segment(p[0])

    @_("literal")
    def iexpr(self, p):
        return NFASegment.create_transition_segment(symbol_str=p[0], caption=p[0])

    @_("LEFT_PAR expr RIGHT_PAR")
    def iexpr(self,p):
        return p[1]
        #return NFASegment.create_transition_segment(symbol_str=f"({p[1]})", caption=f"({p[1]})")
    @_('ANY')
    def iexpr(self, p):
        assert tuple(p) == ('.',)
        return NFASegment.create_transition_segment(symbol_str=ANY_SYMBOL, caption=f"Any")

    @_('WORD')
    def literal(self,p):
        return(p[0].lower())

    @_('CLASS')
    def literal(self,p):
        return(p[0])
        #self.stack.append(p[0])


def compile_regex(regex: str) -> Automaton:
    graph = WordParser().parse_str(regex)
    if graph.start or graph.end:
        raise ValueError(f"Anchors ^ $ are not supported, use a matcher with anchored=True: {regex!r}")
    return graph.to_automaton()


def main():
    while True:
        text = input(">>> ")
        graph = WordParser().parse_str(text)
        #graph.draw_graph(graph.nfa_segment)
        print(graph)
        transitions, starting_states, accepting_states = graph.nfa_segment.create_transition_matrices()
        print("Transitions:")
        for symbol, transition in transitions.items():
            print(symbol,":")
            print(transition)
            print(starting_states)
            print(accepting_states)
            print("")


if __name__ == '__main__':
    main()
