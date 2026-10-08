# Matching: run an automaton over documents (sequences of token ids) given emissions (V, K), the probability that
# vocabulary token v is symbol k; a document scores the probability of its best match. Only tensors: no strings.
# Max-product (Viterbi) semantics: reading token x moves state i to state j with probability
#     max_k transitions[k, i, j] * emissions[x, k],
# and a path's probability is the product along it. In search mode a match may start at any token, so the start
# states are re-activated before every token, and the score is the best probability with which an accept state is
# reached; anchored, the match must span the whole document. With 0/1 emissions this is exactly classic regex matching.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .fa_graph import DFA, Automaton


class AbstractMatcher(object):
    """Scores documents against an automaton: one score per document.

    The base implementation steps through each whole document on its own; subclasses compute the same scores faster
    or implement other semantics.
    """
    def __init__(self, anchored: bool = False, device: Optional[torch.device] = None) -> None:
        # anchored=False: a match may lie anywhere in the document (search). anchored=True: the match must span the
        # whole document, from its first to its last token (as regex fullmatch); an empty document then scores 0.
        self.anchored = anchored
        self.device = device or torch.device("cpu")

    def scores(self, automaton: Automaton, emissions: Tensor, documents: List[Tensor]) -> Tensor:
        transitions = automaton.transitions.to(self.device)
        start, accept = automaton.start.to(self.device), automaton.accept.to(self.device)
        emissions = emissions.to(self.device)
        scores = torch.zeros(len(documents))
        for n, document in enumerate(documents):
            state = (start if self.anchored else torch.zeros_like(start))[None]                 # (1, S)
            best = torch.zeros((), device=self.device)
            for token in document.tolist():
                if not self.anchored:
                    state = torch.maximum(state, start)
                state, _ = self.step(state, emissions[token][None], transitions)
                best = (state[0] * accept).amax() if self.anchored else torch.maximum(best, (state[0] * accept).amax())
            scores[n] = best.cpu()
        return scores

    def matches(self, automaton: Automaton, emissions: Tensor, documents: List[Tensor], min_probability: float = 0.01) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Every match with probability >= min_probability, as (document, start token, end token (exclusive),
        # probability) tensors: for every token a match can end at, the best match ending there. Anchored, a document
        # has at most one match: the whole document.
        transitions = automaton.transitions.to(self.device)
        start, accept = automaton.start.to(self.device), automaton.accept.to(self.device)
        emissions = emissions.to(self.device)
        found = []
        for n, document in enumerate(documents):
            state = start[None].clone() if self.anchored else torch.zeros_like(start)[None]
            origin = torch.zeros_like(state, dtype=torch.long)
            for t, token in enumerate(document.tolist()):
                if not self.anchored:  # a match may start here; an earlier start of the same probability is kept
                    origin = torch.where(start > state, t, origin)
                    state = torch.maximum(state, start)
                state, origin = self.step(state, emissions[token][None], transitions, origin)
                probability, best = (state[0] * accept).max(dim=0)
                if (not self.anchored or t == document.numel() - 1) and probability >= min_probability:
                    found.append((n, int(origin[0, best]), t + 1, float(probability)))
        return _as_tensors(found)

    @staticmethod
    def step(state: Tensor, emission: Tensor, transitions: Tensor, origin: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # state: (B, S), emission: (B, K) of the token read, transitions: (K, S, S) -> next state (B, S).
        # origin: (B, S), optional, the token where each state's best path started; it follows the best paths.
        if origin is None:
            via_symbol = (state[:, None, :, None] * transitions[None]).amax(dim=2)     # (B, K, S): best source per target
            return (via_symbol * emission[:, :, None]).amax(dim=1), None              # (B, S): best symbol per target
        via_symbol, source = (state[:, None, :, None] * transitions[None]).max(dim=2)
        state, symbol = (via_symbol * emission[:, :, None]).max(dim=1)
        source = source.gather(1, symbol[:, None, :])[:, 0]                           # (B, S): best source per target
        return state, origin.gather(1, source)


class ViterbiMatcher(AbstractMatcher):
    """The base scores, computed in parallel over short windows of the documents.

    Only parts of a document that can contain a match are processed: if no match is longer than L tokens and every
    match reads at least one query word, a match lies within L-1 tokens of a token that resembles a query word. Those
    windows ("segments") are short, so all of them run in parallel instead of stepping through whole documents.
    A token resembles a query word if its emission is at least min_emission. Since a path through a token with a lower
    emission scores below min_emission, every score >= min_emission is exact; lower scores may be reported as 0.
    Segments are batched by length within max_segments segments and max_tokens padded tokens.
    """
    def __init__(self, anchored: bool = False, device: Optional[torch.device] = None, min_emission: float = 0.01,
                 max_segments: int = 16384, max_tokens: int = 1 << 22) -> None:
        super().__init__(anchored, device)
        self.min_emission = min_emission
        self.max_segments = max_segments
        self.max_tokens = max_tokens

    def scores(self, automaton: Automaton, emissions: Tensor, documents: List[Tensor]) -> Tensor:
        scores = torch.zeros(len(documents))
        segments, owners, _ = self._segments(automaton, emissions, documents)
        for batch, accepted, _ in self._forward(automaton, emissions, segments, track_origin=False):
            scores.scatter_reduce_(0, owners[batch], accepted.amax(dim=1).cpu(), reduce="amax")
        return scores

    def matches(self, automaton: Automaton, emissions: Tensor, documents: List[Tensor], min_probability: float = 0.01) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # As the base class; exact when min_probability >= min_emission.
        segments, owners, positions = self._segments(automaton, emissions, documents)
        lengths = torch.tensor([segment.numel() for segment in segments], dtype=torch.long)
        found = [_as_tensors([])]
        for batch, accepted, origins in self._forward(automaton, emissions, segments, track_origin=True):
            batch = torch.tensor(batch, dtype=torch.long)
            accepted, origins = accepted.cpu(), origins.cpu()
            inside = torch.arange(accepted.size(1))[None, :] < lengths[batch][:, None]          # not padding
            row, t = ((accepted >= min_probability) & inside).nonzero(as_tuple=True)
            segment = batch[row]
            found.append((owners[segment], positions[segment] + origins[row, t], positions[segment] + t + 1, accepted[row, t]))
        return tuple(torch.cat(column) for column in zip(*found))

    def _segments(self, automaton: Automaton, emissions: Tensor, documents: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        if self.anchored:  # whole documents
            owners = torch.tensor([n for n, document in enumerate(documents) if document.numel() > 0], dtype=torch.long)
            return [documents[n] for n in owners.tolist()], owners, torch.zeros_like(owners)
        return candidate_segments(automaton, emissions, documents, self.min_emission)

    def _forward(self, automaton: Automaton, emissions: Tensor, segments: List[Tensor], track_origin: bool):
        # Yields, batch by batch: the segments' indexes, the probability of the best match ending at each token (B, T)
        # (anchored: only at each segment's last token, 0 elsewhere), and where those matches start (B, T) or None.
        transitions = automaton.transitions.to(self.device)                                    # (K, S, S)
        start, accept = automaton.start.to(self.device), automaton.accept.to(self.device)       # (S,)
        padded_emissions = torch.cat([emissions, torch.zeros(1, emissions.size(1))]).to(self.device)  # last row: padding
        padding = emissions.size(0)
        for batch in batches_by_size([segment.numel() for segment in segments], self.max_segments, self.max_tokens):
            tokens = torch.nn.utils.rnn.pad_sequence([segments[n] for n in batch], batch_first=True, padding_value=padding).to(self.device)
            last = torch.tensor([segments[n].numel() - 1 for n in batch], device=self.device)  # (B,) position of each last token
            state = start.expand(len(batch), -1).clone() if self.anchored else torch.zeros(len(batch), automaton.n_states, device=self.device)  # (B, S)
            origin = torch.zeros_like(state, dtype=torch.long) if track_origin else None
            accepted = torch.zeros(len(batch), tokens.size(1), device=self.device)
            origins = torch.zeros(len(batch), tokens.size(1), dtype=torch.long, device=self.device) if track_origin else None
            for t in range(tokens.size(1)):
                if not self.anchored:                                                       # a match may start here
                    if track_origin:
                        origin = torch.where(start > state, t, origin)
                    state = torch.maximum(state, start)
                state, origin = self.step(state, padded_emissions[tokens[:, t]], transitions, origin)
                probability, best = (state * accept).max(dim=1)
                accepted[:, t] = torch.where(last == t, probability, 0.0) if self.anchored else probability
                if track_origin:
                    origins[:, t] = origin.gather(1, best[:, None])[:, 0]
            yield batch, accepted, origins



class DFAMatcher(ViterbiMatcher):
    """Total-probability matching on the determinized automaton.

    A token's reading is categorical: it is exactly one of the query words, or none of them (the emission rows of
    ExactObserver and JointSoftmaxObserver; observers whose word probabilities sum above 1 are rejected), while ANY and
    token classes are certain. The DFA's letters are these readings; its states, sets of automaton states. Since a DFA
    is in exactly one state, the forward pass keeps a distribution over its states, and the mass on accepting states
    after token t is the probability that a match ends at t (sum-product over all readings, where Viterbi keeps the
    best path). A document scores its most probable end; a match starts where the reversed DFA, read backwards from
    the end, finds the most probable span. With 0/1 emissions this is classic matching, as Viterbi.
    Windows and batches as ViterbiMatcher; the DFA is built eagerly (up to max_states states) and minimized.
    """
    def __init__(self, anchored: bool = False, device: Optional[torch.device] = None, min_emission: float = 0.01,
                 max_segments: int = 16384, max_tokens: int = 1 << 22, max_states: int = 1 << 16) -> None:
        super().__init__(anchored, device, min_emission, max_segments, max_tokens)
        self.max_states = max_states

    def matches(self, automaton: Automaton, emissions: Tensor, documents: List[Tensor], min_probability: float = 0.01) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        segments, owners, positions = self._segments(automaton, emissions, documents)
        lengths = torch.tensor([segment.numel() for segment in segments], dtype=torch.long)
        found = [_as_tensors([])]
        for batch, accepted, _ in self._forward(automaton, emissions, segments, track_origin=False):
            batch = torch.tensor(batch, dtype=torch.long)
            accepted = accepted.cpu()
            inside = torch.arange(accepted.size(1))[None, :] < lengths[batch][:, None]
            row, t = ((accepted >= min_probability) & inside).nonzero(as_tuple=True)
            segment = batch[row]
            if self.anchored:
                starts = torch.zeros_like(t)
            else:
                starts = self._starts(automaton, emissions, [segments[s][:end + 1] for s, end in zip(segment.tolist(), t.tolist())])
            found.append((owners[segment], positions[segment] + starts, positions[segment] + t + 1, accepted[row, t]))
        return tuple(torch.cat(column) for column in zip(*found))

    def _starts(self, automaton: Automaton, emissions: Tensor, prefixes: List[Tensor]) -> Tensor:
        # For each token sequence ending at a match end: where the most probable match ending there starts (the
        # longest one on ties), read backwards on the reversed DFA.
        longest = longest_match(automaton)
        windows = [prefix.flip(0)[:longest] if longest is not None else prefix.flip(0) for prefix in prefixes]
        reverse = self._determinize(automaton.reversed(), emissions, search=False)
        starts = torch.zeros(len(windows), dtype=torch.long)
        for batch in batches_by_size([window.numel() for window in windows], self.max_segments, self.max_tokens):
            tokens = torch.nn.utils.rnn.pad_sequence([windows[n] for n in batch], batch_first=True, padding_value=emissions.size(0)).to(self.device)
            accepted = self._dfa_forward(reverse, tokens).cpu()                                    # (B, L)
            best = accepted.amax(dim=1, keepdim=True)
            length = (accepted >= best * (1 - 1e-6)).float().cumsum(1).argmax(dim=1)              # last maximum: the longest
            starts[torch.tensor(batch)] = torch.tensor([prefixes[n].numel() - 1 for n in batch]) - length
        return starts

    def _forward(self, automaton: Automaton, emissions: Tensor, segments: List[Tensor], track_origin: bool):
        dfa = self._determinize(automaton, emissions, search=not self.anchored)
        for batch in batches_by_size([segment.numel() for segment in segments], self.max_segments, self.max_tokens):
            tokens = torch.nn.utils.rnn.pad_sequence([segments[n] for n in batch], batch_first=True, padding_value=emissions.size(0)).to(self.device)
            accepted = self._dfa_forward(dfa, tokens)
            if self.anchored:  # only complete documents count
                last = torch.tensor([segments[n].numel() - 1 for n in batch], device=self.device)
                accepted = torch.where(torch.arange(tokens.size(1), device=self.device)[None, :] == last[:, None], accepted, 0.0)
            yield batch, accepted, None

    def _dfa_forward(self, determinized: Tuple[DFA, Tensor, Tensor], tokens: Tensor) -> Tensor:
        # tokens: (B, T) vocabulary ids, padding = V. Returns the probability mass on accepting states after each token.
        dfa, readings, pattern = determinized
        delta, accepting = dfa.transitions.to(self.device), dfa.accept.to(self.device)
        n_words = readings.size(1)
        probability = torch.zeros(tokens.size(0), dfa.n_states, device=self.device)                # (B, D)
        probability[:, dfa.initial] = 1
        accepted = torch.zeros(tokens.shape, device=self.device)
        for t in range(tokens.size(1)):
            token = tokens[:, t]
            letters = pattern[token] * n_words                                                     # (B,) first letter of each token's pattern
            moved = torch.zeros_like(probability)
            for word in range(n_words):  # reading "none" (word 0) or query word `word`
                destination = delta[:, letters + word].T                                           # (B, D)
                moved.scatter_add_(1, destination, probability * readings[token, word][:, None])
            probability = moved
            accepted[:, t] = probability[:, accepting].sum(dim=1)
        return accepted

    def _determinize(self, automaton: Automaton, emissions: Tensor, search: bool) -> Tuple[DFA, Tensor, Tensor]:
        # The letters are (class pattern, reading): the 0/1 values of the token-class columns that occur in the
        # vocabulary, times "none" or one of the W query words. Returns the minimized DFA over them and, per vocabulary
        # token (+ a padding row), the probabilities of its readings (V + 1, W + 1) and its class pattern (V + 1,): the
        # token's letters are pattern * (W + 1) + reading.
        words, classes = automaton.word_mask, ~automaton.word_mask & ~automaton.any_mask
        word_emissions = emissions[:, words]
        if (word_emissions.sum(dim=1) > 1 + 1e-5).any():
            raise ValueError("DFAMatcher needs categorical readings: a token's query-word probabilities must sum to at most 1")
        class_emissions = emissions[:, classes]
        if ((class_emissions != 0) & (class_emissions != 1)).any():
            raise ValueError("DFAMatcher needs 0/1 emissions for token classes")
        if class_emissions.numel():
            patterns, pattern = torch.unique(class_emissions > 0.5, dim=0, return_inverse=True)   # (P, C), (V,)
        else:  # no token classes (or an empty vocabulary): a single pattern
            patterns, pattern = torch.zeros(1, class_emissions.size(1), dtype=torch.bool), torch.zeros(emissions.size(0), dtype=torch.long)
        readings = torch.cat([(1 - word_emissions.sum(dim=1, keepdim=True)).clamp(min=0), word_emissions], dim=1)
        readings = torch.cat([readings, torch.eye(1, readings.size(1))])                           # padding: reads "none"
        pattern = torch.cat([pattern.reshape(-1), torch.zeros(1, dtype=torch.long)])

        n_readings, n_symbols = int(words.sum()) + 1, len(automaton.symbols)
        letters = torch.zeros(patterns.size(0), n_readings, n_symbols, dtype=torch.bool)
        letters[:, :, automaton.any_mask] = True
        letters[:, :, classes] = patterns[:, None, :]
        letters[:, torch.arange(1, n_readings), words.nonzero().reshape(-1)] = True
        dfa = automaton.determinize(letters.reshape(-1, n_symbols), search=search, max_states=self.max_states).minimized()
        return dfa, readings.to(self.device), pattern.to(self.device)


def longest_match(automaton: Automaton) -> Optional[int]:
    # The most tokens any match can read, or None if unbounded (a cycle) or if a match can read only ANY tokens.
    any_mask = automaton.any_mask
    step = automaton.transitions.amax(dim=0)  # (S, S): reading some token can lead from i to j
    any_step = automaton.transitions[any_mask].amax(dim=0) if any_mask.any() else torch.zeros_like(step)
    if any_mask.all() or _reaches(automaton.start, any_step, automaton.accept, automaton.n_states):
        return None
    reached, longest = automaton.start, 0
    for length in range(1, automaton.n_states + 1):
        reached = ((reached @ step) > 0).float()
        if not reached.any():
            return longest
        if (reached * automaton.accept).any():
            longest = length
    return None  # still reaching states after more steps than there are states: there is a cycle


def _reaches(start: Tensor, step: Tensor, accept: Tensor, n_states: int) -> bool:
    reached = start
    for _ in range(n_states):
        reached = ((reached @ step) > 0).float()
        if (reached * accept).any():
            return True
    return False


def _as_tensors(found: List[Tuple[int, int, int, float]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    columns = list(zip(*found)) if found else [(), (), (), ()]
    return (torch.tensor(columns[0], dtype=torch.long), torch.tensor(columns[1], dtype=torch.long),
            torch.tensor(columns[2], dtype=torch.long), torch.tensor(columns[3], dtype=torch.float))


def candidate_segments(automaton: Automaton, emissions: Tensor, documents: List[Tensor], min_emission: float) -> Tuple[List[Tensor], Tensor, Tensor]:
    # Returns the token segments worth processing and, for each, the index of the document it comes from and the
    # position of its first token in that document.
    # Tokens whose emissions are all below min_emission are treated as unable to be a query word.
    words = ~automaton.any_mask
    resembles = emissions[:, words].amax(dim=1) >= min_emission if words.any() else torch.ones(emissions.size(0), dtype=torch.bool)
    lengths = torch.tensor([document.numel() for document in documents], dtype=torch.long)
    if lengths.sum() == 0:
        return [], torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    tokens = torch.cat(documents)
    owner = torch.repeat_interleave(torch.arange(len(documents)), lengths)
    longest = longest_match(automaton)
    if longest is None:  # whole documents that contain at least one resembling token
        keep = torch.zeros(len(documents), dtype=torch.bool).index_fill_(0, owner[resembles[tokens]], True)[owner]
    else:  # every token within longest-1 positions of a resembling token
        radius = longest - 1
        keep = torch.nn.functional.max_pool1d(resembles[tokens].float()[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0] > 0
    # A segment is a run of kept tokens within one document.
    starts = keep.clone()
    starts[1:] &= ~keep[:-1] | (owner[1:] != owner[:-1])
    kept_positions = keep.nonzero().reshape(-1)
    run = torch.cumsum(starts.long(), 0)[kept_positions] - 1
    n_runs = int(starts.sum())
    segment_lengths = torch.bincount(run, minlength=n_runs)
    segments = list(torch.split(tokens[kept_positions], segment_lengths.tolist()))
    # Windows spill across document boundaries; a spilled run without any resembling token cannot match.
    useful = torch.bincount(run, weights=resembles[tokens[kept_positions]].float(), minlength=n_runs) > 0
    owners = owner[starts]
    positions = (torch.arange(tokens.numel()) - torch.cumsum(lengths, 0)[owner] + lengths[owner])[starts]
    return [segment for segment, keep_it in zip(segments, useful.tolist()) if keep_it], owners[useful], positions[useful]


def batches_by_size(lengths: List[int], max_segments: int, max_tokens: int) -> List[List[int]]:
    # Segments sorted by length, grouped so that a batch never exceeds max_segments, nor max_tokens once padded to its
    # longest segment (a single longer segment gets a batch of its own).
    order = sorted(range(len(lengths)), key=lambda n: lengths[n])
    batches: List[List[int]] = []
    for n in order:
        if batches and len(batches[-1]) < max_segments and (len(batches[-1]) + 1) * lengths[n] <= max_tokens:
            batches[-1].append(n)
        else:
            batches.append([n])
    return batches
