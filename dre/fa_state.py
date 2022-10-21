from collections import defaultdict


class FAState(object):
    __slots__ = 'visual_caption', 'id', '_incoming', '_outgoing', 'is_starting', 'is_ending'
    def __init__(self,visual_caption="NA"):
        #self.is_accepting=False
        self.visual_caption=visual_caption
        self.id=None
        self._incoming = []
        self._outgoing = []
        self.is_starting = None
        self.is_ending = None

    def add_incoming(self,transition):
        self._incoming.append(transition)

    def add_outgoing(self,transition):
        self._outgoing.append(transition)

    def remove_incoming(self,state_or_transition,tolerate_non_existing=False):
        if isinstance(state_or_transition,FAState):
            state=state_or_transition
            new_incoming=[i for i in self._incoming if i.source is not state]
            if not tolerate_non_existing and len(new_incoming)==len(self._incoming):
                raise ValueError
            self._incoming=new_incoming
        elif isinstance(state_or_transition,FATransition):
            transition=state_or_transition
            new_incoming=[i for i in self._incoming if i is not transition]
            if not tolerate_non_existing and len(new_incoming)==len(self._incoming):
                raise ValueError
            self._incoming=new_incoming

    def remove_outgoing(self,state_or_transition,tolerate_non_existing=True):
        if isinstance(state_or_transition,FAState):
            state = state_or_transition
            new_outgoing=[i for i in self._outgoing if i.destination is not state]
            if not tolerate_non_existing or len(new_outgoing)==len(self._outgoing):
                raise ValueError
            self._outgoing=new_outgoing
        elif isinstance(state_or_transition,FATransition):
            transition = state_or_transition
            new_outgoing=[i for i in self._outgoing if i is not transition]
            if not tolerate_non_existing and len(new_outgoing)==len(self._outgoing):
                raise ValueError
            self._outgoing=new_outgoing

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
        res=FAState(visual_caption=self.visual_caption)
        res.is_ending=self.is_ending
        res.is_starting=self.is_starting
        res.id=self.id
        return res

    def get_descendant_count(self):
        return len(self.get_descendants())

    def get_descendants(self):
        visited=set()
        self._get_descendants(visited)
        return list(visited)

    def _get_descendants(self,visited):
        children=[tr.destination for tr in self._outgoing]
        for child in children:
            if child not in visited:
                visited.add(child)
                child._get_descendants(visited)

    def get_children_and_null_descendants(self):
        children_dict={}
        for transition in self.outgoing_transitions:
            if bool(transition):
                children_dict[transition.symbol_str]=transition.destination
        children_dict[None]=[self]+self._get_null_descendants([])
        return children_dict

    def get_non_null_children_and_their_null_descendants(self):
        """Returns all nodes that are a single symbol away from self.

        e*s union s*e
        :return:
        """
        children_dict=defaultdict(lambda:[])
        for transition in self.outgoing_transitions:
            if bool(transition): # symbol following Null
                children=transition.destination._get_null_descendants([])
                if transition.destination not in children:
                    children.append(transition.destination)
                children_dict[transition.symbol_str]+=(children)

        null_descendants=self._get_null_descendants([])
        for null_child in null_descendants:
            for transition in null_child._outgoing:
                if transition: # A non-null child of a null children
                    children_dict[transition.symbol_str].append(transition.destination)
        children_dict={k:list(sorted(set(v))) for k,v in list(children_dict.items())}
        return children_dict

    def _get_null_descendants(self,visited):
        result=[]
        for transition in self.outgoing_transitions:
            if (transition.destination not in visited) and not bool(transition):
                result.append(transition.destination)
                transition.destination._get_null_descendants(visited+[self])
        return result

    @property
    def is_removable(self):
        if len(self._incoming)<1 or len(self._outgoing)<1:
            #starting or accepting
            return False

        if len(self._incoming)>1 or len(self._outgoing)>1:
            #multiple incoming or outgoing
            return False

        if (bool(self._incoming[0]) and bool(self._outgoing[0])):
            # both incoming and outgoig not epsilons
            return False
        return True

    def remove(self):
        raise Exception
        if len(self._outgoing)==1 and not(self._outgoing[0]):
            replacing_state=self._outgoing[0].destination
            for tr in self._incoming:
                tr.set_destination(replacing_state)
            replacing_state.remove_incoming(self)
        elif len(self._incoming)==1 and not(self._incoming[0]):
            replacing_state=self._incoming[0].source
            for tr in self._outgoing:
                tr.set_source(replacing_state)
            replacing_state.remove_outgoing(self)
        else:
            raise ValueError

    def __hash__(self):
        return object.__hash__(self)

    def __str__(self):
        id=self.id
        start_str=">" if self.is_starting else " "
        end_str = "<" if self.is_ending else " "
        if id is None:
            id=-1
        return "%sS{%d|%d>%d}%s"%(start_str,id,len(self.incoming_transitions),len(self.outgoing_transitions),end_str)
    
    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self,other):
        python_id_cmp=.5*(id(self)<id(other))
        s_id = self.id
        o_id = other.id
        if self.id is None:
            s_id=-1
        if other.id is None:
            o_id=-1
        return s_id+python_id_cmp<o_id
