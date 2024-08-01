class FAState(object):
    __slots__ = 'visual_caption', 'id', '_incoming', '_outgoing', 'is_starting', 'is_ending'

    def __init__(self, visual_caption="NA"):
        self.visual_caption = visual_caption
        self.id = None
        self._incoming = []
        self._outgoing = []
        self.is_starting = None
        self.is_ending = None

    @property
    def incoming_states(self):
        return tuple([transition.source for transition in self._incoming])

    @property
    def incoming_transitions(self):
        return tuple(self._incoming)

    @property
    def outgoing_states(self):
        return tuple([transition.destination for transition in self._outgoing])

    @property
    def outgoing_transitions(self):
        return tuple(self._outgoing)

    def clone_disconnected(self):
        res = FAState(visual_caption=self.visual_caption)
        res.is_ending = self.is_ending
        res.is_starting = self.is_starting
        res.id = self.id
        return res

    def get_descendant_count(self):
        return len(self.get_descendants())

    def get_descendants(self):
        visited = set()
        self._get_descendants(visited)
        return list(visited)

    def _get_descendants(self, visited):
        children = [tr.destination for tr in self._outgoing]
        for child in children:
            if child not in visited:
                visited.add(child)
                child._get_descendants(visited)

    def __hash__(self):
        return object.__hash__(self)

    def __str__(self):
        node_id = self.id
        start_str = ">" if self.is_starting else " "
        end_str = "<" if self.is_ending else " "
        if node_id is None:
            node_id = -1
        node_str = f"S{'{'}{node_id}|{len(self.incoming_transitions)}>{len(self.outgoing_transitions)}{'}'}"
        return f"{start_str}{node_str}{end_str}"

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        python_id_cmp = .5*(id(self) < id(other))
        s_id = self.id
        o_id = other.id
        if self.id is None:
            s_id = -1
        if other.id is None:
            o_id = -1
        return s_id + python_id_cmp < o_id
