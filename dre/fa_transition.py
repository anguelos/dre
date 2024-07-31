from dre.fa_state import FAState


class FATransition(object):
    def __init__(self, symbol_str="", caption="", source=None, destination=None):
        assert source is not None
        assert destination is not None
        if symbol_str == "" and caption=="":
            caption = "ε"
        self.symbol_str=symbol_str
        self.caption=caption
        self._source = source
        self._destination = destination
        if source is not None:
            source._outgoing.append(self)
        if destination is not None:
            destination._incoming.append(self)

    @property
    def src_id(self):
        if self._source:
            return self._source.id
        else:
            return -2

    @property
    def dst_id(self):
        if self._source:
            return self._destination.id
        else:
            return -2

    @property
    def source(self):
        return self._source

    @property
    def destination(self):
        return self._destination

    # def set_source(self,state):
    #     raise Exception
    #     if self._source is not None:
    #         self._source._outgoing.remove(self)
    #     self._source=state
    #     if self not in state._outgoing:
    #         state._outgoing.append(self)

    # def set_destination(self,state):
    #     raise Exception
    #     if self._destination is not None:
    #         self._destination._incoming.remove(self)
    #     self._destination=state
    #     if self not in state._incoming:
    #         state._incoming.append(self)

    # def remove(self):
    #     if self._destination is not None:
    #         self._destination._incoming.remove(self)
    #     if self._source is not None:
    #         self._source._outgoing.remove(self)
    #     return
    #     self._destination
    #     dst=self._destination
    #     dst._incoming=[tr for tr in dst._incoming if tr is not self]
    #     src=self._source
    #     src._outgoing=[tr for tr in src._outgoing if tr is not self]
    #     self._destination=None
    #     self._source=None

    def __str__(self):
        #print("DBG:",repr((self.src_id, self.caption, self.dst_id)))
        #return "%2d->T{%3s}->%2d"%(self.src_id, self.caption, self.dst_id)
        return f"{self.src_id}->T|{self.caption}| {self.dst_id}"

    def __bool__ (self):
        return len(self.symbol_str) > 0
    
    def __hash__(self):
        # TODO(anguelos) find a better hash
        if self._source is not None and self._destination is not None:
            return hash(self._source)*hash(self._destination)*hash(self.symbol_str)
        else:
            raise ValueError
    
    #def __del__(self):
    #    print("__del__{")
    #    self.remove()
    #    print("}__del__")

    #def __eq__(self, other):
    #    if not isinstance(other, FAState):
    #        return False
    #    if self._source is None or self._destination is None:
    #        return False
    #    return (self._source, self._destination, self.symbol_str) == (other._source, other._destination, other.symbol_str)

    def __lt__(self, other):
        return str(self)<str(other)
    def __eq__(self,other):
        return str(self)==str(other)


    # def clone(self,source=None,destination=None):
    #     raise Exception
    #     res=FATransition(symbol_str=self.symbol_str, caption=self.caption)
    #     res._source=source
    #     res._destination=destination
    #     return res

    # @classmethod
    # def create_transition(cls, symbol_str="",caption="\\u2205"):
    #     return cls(symbol_str=symbol_str,caption=caption)
