"""IT5005 Assignment 1: student implementation file.

Implement the functions marked below. Do not modify utils.py or logic_.py.
"""

from utils import *
from logic_ import *


# Do not change this function; it is used to create atomic propositions.
def atom(prefix, r, c, v):
    """prefix is 'Is' or 'Not'. Returns the Expr for e.g. Is3_2_4."""
    return expr(f'{prefix}{r}_{c}_{v}')

def build_general_kb(n, box_h, box_w, givens):
    """Return a PropKB encoding this n x n Sudoku's constraints plus the given
    cells, as general clauses.
    """
    kb = PropKB()
    cells = [(r, c) for r in range(1, n + 1) for c in range(1, n + 1)]

    def peers_of(r, c):
        peers = set()
        for c2 in range(1, n + 1):
            if c2 != c:
                peers.add((r, c2))
        for r2 in range(1, n + 1):
            if r2 != r:
                peers.add((r2, c))
        br, bc = (r - 1) // box_h, (c - 1) // box_w
        for r2 in range(br * box_h + 1, br * box_h + box_h + 1):
            for c2 in range(bc * box_w + 1, bc * box_w + box_w + 1):
                if (r2, c2) != (r, c):
                    peers.add((r2, c2))
        return peers

    # At least one value per cell
    for (r, c) in cells:
        lits = [atom('Is', r, c, v) for v in range(1, n + 1)]
        kb.tell(associate('|', lits))

    # At most one value per cell
    for (r, c) in cells:
        for v1 in range(1, n + 1):
            for v2 in range(v1 + 1, n + 1):
                kb.tell(~atom('Is', r, c, v1) | ~atom('Is', r, c, v2))

    # No two peer cells share a value
    for cell in cells:
        r, c = cell
        for peer in peers_of(r, c):
            if peer > cell:
                pr, pc = peer
                for v in range(1, n + 1):
                    kb.tell(~atom('Is', r, c, v) | ~atom('Is', pr, pc, v))

    # Givens
    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    return kb

def build_definite_kb(n, box_h, box_w, givens):
    """Return a PropDefiniteKB encoding this n x n Sudoku's constraints plus
    the given cells, using elimination + last-candidate reasoning.
    """
    kb = PropDefiniteKB()
    cells = [(r, c) for r in range(1, n + 1) for c in range(1, n + 1)]

    def peers_of(r, c):
        peers = set()
        for c2 in range(1, n + 1):
            if c2 != c:
                peers.add((r, c2))
        for r2 in range(1, n + 1):
            if r2 != r:
                peers.add((r2, c))
        br, bc = (r - 1) // box_h, (c - 1) // box_w
        for r2 in range(br * box_h + 1, br * box_h + box_h + 1):
            for c2 in range(bc * box_w + 1, bc * box_w + box_w + 1):
                if (r2, c2) != (r, c):
                    peers.add((r2, c2))
        return peers

    # Elimination: peer holds v ==> this cell is Not v
    for (r, c) in cells:
        for (pr, pc) in peers_of(r, c):
            for v in range(1, n + 1):
                kb.tell(atom('Is', pr, pc, v) |'==>'| atom('Not', r, c, v))

    # Last candidate: every other value eliminated ==> this cell is v_target
    for (r, c) in cells:
        for v_target in range(1, n + 1):
            premises = [atom('Not', r, c, v) for v in range(1, n + 1) if v != v_target]
            kb.tell(associate('&', premises) |'==>'| atom('Is', r, c, v_target))

    # Givens
    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    return kb

def solve_full_grid_fc(n, box_h, box_w, givens):
    """Solve the full n x n Sudoku grid using forward chaining.

    Parameters
    ----------
    n : int
        The size of the grid (n x n).
    box_h, box_w : int
        The height and width of each box region.
    givens : dict
        Mapping (row, col) -> value for the cells specified by the puzzle.

    Returns
    -------
    dict
        Mapping (row, col) -> value for every cell in the solved grid.
    """
    kb = build_definite_kb(n, box_h, box_w, givens)
    solved = {}

    def peers_of(r, c):
        peers = set()
        for c2 in range(1, n + 1):
            if c2 != c:
                peers.add((r, c2))
        for r2 in range(1, n + 1):
            if r2 != r:
                peers.add((r2, c))
        br, bc = (r - 1) // box_h, (c - 1) // box_w
        for r2 in range(br * box_h + 1, br * box_h + box_h + 1):
            for c2 in range(bc * box_w + 1, bc * box_w + box_w + 1):
                if (r2, c2) != (r, c):
                    peers.add((r2, c2))
        return peers

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            # Already known directly from the puzzle -- no need to
            # spend an expensive pl_fc_entails call re-deriving it
            if (r, c) in givens:
                solved[(r, c)] = givens[(r, c)]
                continue
            # Values already ruled out by a peer's GIVEN value need no
            # expensive pl_fc_entails call -- basic Sudoku uniqueness
            # already guarantees they can't be this cell's value.
            peer_given_values = {givens[p] for p in peers_of(r, c) if p in givens}
            candidates = [v for v in range(1, n + 1) if v not in peer_given_values]
            for v in candidates:
                if pl_fc_entails(kb, atom('Is', r, c, v)):
                    solved[(r, c)] = v
                    break
    return solved



def pl_bc_entails(kb, query):
    """Determine whether a definite-clause KB entails a query, using
    backward chaining.

    Parameters
    ----------
    kb : PropDefiniteKB
        The knowledge base of definite clauses.
    query : Expr
        The atomic proposition to prove.

    Returns
    -------
    bool
        True if kb entails query, False otherwise.
    """

    # Permanent memory — only ever holds proven-True facts, and is
    # kept across every retry round below
    permanent_cache = {}

    # One single search attempt. 'round_memo' is a scratch memory
    # just for THIS attempt — it remembers both True and False
    # answers so nothing gets recomputed twice within this one round,
    # but it gets thrown away and rebuilt fresh next round.
    def bc_once(q, visiting, round_memo):
        if q in permanent_cache:
            return True
        if q in round_memo:
            return round_memo[q]
        if q in visiting:
            return False

        visiting.add(q)
        result = False

        for clause in kb.clauses:
            if is_symbol(clause.op) and clause == q:
                result = True
                break

        if not result:
            for clause in kb.clauses:
                if clause.op == '==>':
                    premise, conclusion = clause.args
                    if conclusion == q:
                        premises = conjuncts(premise)
                        if all(bc_once(p, visiting, round_memo) for p in premises):
                            result = True
                            break

        visiting.discard(q)
        round_memo[q] = result
        if result:
            permanent_cache[q] = True
        return result

    while True:
        before = len(permanent_cache)
        found = bc_once(query, set(), {})   # fresh round_memo every round
        if found or len(permanent_cache) == before:
            return found

def solve_full_grid_bc(n, box_h, box_w, givens):
    """Solve the full n x n Sudoku grid using backward chaining.

    Parameters
    ----------
    n : int
        The size of the grid (n x n).
    box_h, box_w : int
        The height and width of each box region.
    givens : dict
        Mapping (row, col) -> value for the cells specified by the puzzle.

    Returns
    -------
    dict
        Mapping (row, col) -> value for every cell in the solved grid.
    """
    kb = build_definite_kb(n, box_h, box_w, givens)
    solved = {}

    def peers_of(r, c):
        peers = set()
        for c2 in range(1, n + 1):
            if c2 != c:
                peers.add((r, c2))
        for r2 in range(1, n + 1):
            if r2 != r:
                peers.add((r2, c))
        br, bc = (r - 1) // box_h, (c - 1) // box_w
        for r2 in range(br * box_h + 1, br * box_h + box_h + 1):
            for c2 in range(bc * box_w + 1, bc * box_w + box_w + 1):
                if (r2, c2) != (r, c):
                    peers.add((r2, c2))
        return peers

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            # Same idea here -- skip the expensive search entirely
            # for a cell whose value is already given
            if (r, c) in givens:
                solved[(r, c)] = givens[(r, c)]
                continue
            # Same peer-given pruning as the FC version -- skip values
            # basic Sudoku uniqueness already rules out.
            peer_given_values = {givens[p] for p in peers_of(r, c) if p in givens}
            candidates = [v for v in range(1, n + 1) if v not in peer_given_values]
            for v in candidates:
                if pl_bc_entails(kb, atom('Is', r, c, v)):
                    solved[(r, c)] = v
                    break
    return solved
