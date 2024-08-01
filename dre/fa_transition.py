from dre.fa_state import FAState


class FATransition:
    """Finit automata transition class.

    The class is a directed edge in the graph representation of a finite automata.
    Essentially, it is a tuple of the form (source, destination, symbol_str).
    """
    def __init__(self, symbol_str="", caption="", source=None, destination=None):
        assert source is not None
        assert destination is not None
        if symbol_str == "" and caption == "":
            caption = "ε"
        self.symbol_str = symbol_str
        self.caption = caption
        self._source = source
        self._destination = destination
        if source is not None:
            source._outgoing.append(self)
        if destination is not None:
            destination._incoming.append(self)

    @property
    def src_id(self):
        """id getter for the source state.

        Returns:
            int: the id of the source state.
        """
        if self._source:
            return self._source.id
        else:
            return -2

    @property
    def dst_id(self):
        """id getter for the destination state.

        Returns:
            int: the id of the destination state.
        """
        if self._source:
            return self._destination.id
        else:
            return -2

    @property
    def source(self):
        """Source getter.

        Returns:
            FAState: the source state of the transition.
        """
        return self._source

    @property
    def destination(self):
        """Destiantion getter.

        Returns:
            FAState: the destination state of the transition.
        """
        return self._destination

    def __str__(self):
        return f"{self.src_id}->T|{self.caption}| {self.dst_id}"

    def __bool__(self):
        return len(self.symbol_str) > 0

    def __hash__(self):
        # TODO(anguelos) find a better hash
        if self._source is not None and self._destination is not None:
            return hash(self._source)*hash(self._destination)*hash(self.symbol_str)
        raise ValueError("Transition is not connected to a state")

    def __lt__(self, other):
        return str(self) < str(other)

    def __eq__(self, other):
        return str(self) == str(other)
