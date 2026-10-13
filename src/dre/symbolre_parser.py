import sys
sys.path.append("./dre")
from abstract_nfa import FAGraph


from .abstract_parser import Lexer,  AbstractParser
from abstract_nfa import FAGraph
from collections import defaultdict



class SymbolLexer(Lexer):
    # This is the set of tokens we are exporting to the Parser
    tokens = {ANY, START, END, SYMBOL, ASTERISK, QUESTIONMARK, PLUS, OR, LEFT_PAR, RIGHT_PAR}
    # Any literals we want to ignore
    ignore = ' \t'
    # Any literals we did not define as tokens, will be available for usage in the Parser
    literals = {'.', '!'}

    # The definition of each token in a regex pattern - Notice that the order MATTERS!! First match will be taken
    ANY = r'\.'
    START = r'\^'
    END = r'\$'
    SYMBOL = r'[A-Za-z0-9]|\\\.|!|_|,|;|:|,|\\\?|\\-|\\\^|\\\$|\\\\'
    ASTERISK = r'\*'
    QUESTIONMARK = r'\?'
    PLUS = r'\+'
    OR = r'\|'
    LEFT_PAR = r'\('
    RIGHT_PAR = r'\)'


class SymbolParser(AbstractParser):
    tokens = SymbolLexer.tokens

    # Un-comment the following line to output the parser logs for debugging any conflicts
    debugfile = 'parser.out'

    # This sets the order of execution. The last values will have a higher precedence
    precedence = (
        ('left', ASTERISK, QUESTIONMARK, PLUS),
        ('left', OR),
        ('left', RIGHT_PAR),
        ('right', LEFT_PAR),
    )

    def __init__(self):
        self.lexer = SymbolLexer()
    
    def parse_str(self, text):
        tokens = self.lexer.tokenize(text) # Creates a generator of tokens
        result = self.parse(tokens)
        result.re_str = text
        return result

    @property
    def last_item_on_stack(self):
        return self.stack[-1] if len(self.stack) > 0 else None

    @_('START expr END')
    def regex(self, p):
        print('START+ expr END+')
        segment = p[1]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=True, end=True)

    @_('START expr')
    def regex(self, p):
        print('START+ expr END-')
        segment = p[1]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=True, end=False)

    @_('expr END')
    def regex(self, p):
        print('START- expr END+')
        segment = p[0]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=False, end=True)

    @_('expr')
    def regex(self, p):
        print('START- expr END-')
        segment = p[0]
        assert isinstance(segment, NFASegment)        
        return FAGraph(segment=segment, start=False, end=False)

    @_("orexpr","literal", "iexpr", "uexpr")
    def expr(self, p):
        #print ("EXPR", tuple(p))
        #return NFASegment.create_transition_segment(symbol_str=p[0], caption=p[0])
        if isinstance(p[0], NFASegment):
            return p[0]
        elif isinstance(p[0], str):
            return NFASegment.create_transition_segment(symbol_str=p[0], caption=f"expr:{p[0]}")
        else:
            raise ValueError

    @_("expr expr")
    def expr(self, p):
        print("E_E", tuple(p))
        return NFASegment.create_concatenate_segment(p[0], p[1])
        

    @_("expr OR expr")
    def orexpr(self, p):
        print("OR", tuple(p))
        return NFASegment.create_or_segment(p[0], p[2])

    @_("expr ASTERISK")
    def uexpr(self, p):
        print("ASTERISK", tuple(p))
        return NFASegment.create_asterisk_segment(p[0])

    @_("expr QUESTIONMARK")
    def uexpr(self, p):
        print ("QUESTIONMARK", tuple(p))
        return NFASegment.create_question_segment(p[0])

    @_("expr PLUS")
    def uexpr(self,p):
        print("PLUS", tuple(p))
        return NFASegment.create_plus_segment(p[0])

    @_("literal")
    def iexpr(self, p):
        print ("LITERAL", tuple(p))
        return NFASegment.create_transition_segment(symbol_str=p[0], caption=p[0])

    @_("LEFT_PAR expr RIGHT_PAR")
    def iexpr(self,p):
        print(f"Parenthesis:{repr(p)}")
        return NFASegment.create_transition_segment(symbol_str=f"({p[1]})", caption=f"({p[1]})")
    @_('ANY')
    def iexpr(self, p):
        print(f"p:{repr(tuple(p))}")
        assert tuple(p) == ('.',)
        return NFASegment.create_transition_segment(symbol_str=WordLexer.ANY, caption=f"Any")

    @_('WORD')
    def literal(self,p):
        print(f"LITERAL:{repr(p[0])}")
        return(p[0])
        #self.stack.append(p[0])


if __name__ == '__main__':
    while True:
        text = input(">>> ")
        graph = SymbolParser().parse_str(text)
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
