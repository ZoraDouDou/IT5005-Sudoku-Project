import json
import time
from collections import deque
import streamlit as st
from utils import *
from logic_ import *
from sudoku_solver import (
    atom,
    build_definite_kb,
    build_general_kb,
    solve_full_grid_fc,
    solve_full_grid_bc,
    pl_bc_entails,
)

st.set_page_config(page_title='Sudoku Solver', layout='centered')

st.title('Sudoku Solver')

with open('puzzles.json') as f:
    pool = json.load(f)


def parse_givens(puzzle):
    """Convert JSON's 'r_c' keys to the solver's (row, column) keys."""
    return {
        tuple(int(part) for part in key.split('_')): value
        for key, value in puzzle['givens'].items()
    }


def board_html(n, box_h, box_w, givens, solved=None):
    """Render a read-only board, distinguishing givens from solved cells."""
    cells = []
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            given = (r, c) in givens
            value = givens.get((r, c)) if given else (solved or {}).get((r, c))
            cell_kind = 'given' if given else ('filled' if value is not None else 'empty')
            right = '3px solid #344054' if c % box_w == 0 and c != n else '1px solid #cbd5e1'
            bottom = '3px solid #344054' if r % box_h == 0 and r != n else '1px solid #cbd5e1'
            label = (
                f'Row {r}, column {c}: given {value}' if given else
                f'Row {r}, column {c}: solved as {value}' if value is not None else
                f'Row {r}, column {c}: empty'
            )
            cells.append(
                f'<div role="gridcell" aria-label="{label}" class="sudoku-cell '
                f'{cell_kind}" '
                f'style="border-right:{right};border-bottom:{bottom}">'
                f'{value if value is not None else "&nbsp;"}</div>'
            )

    return (
        '<style>'
        '.sudoku-board{display:grid;max-width:540px;'
        f'grid-template-columns:repeat({n},minmax(0,1fr));'
        'border:3px solid #344054;}'
        '.sudoku-cell{aspect-ratio:1;display:flex;align-items:center;'
        'justify-content:center;font-size:clamp(1rem,3vw,1.65rem);}'
        '.sudoku-cell.given{background:#e8f0fb;color:#142d4e;font-weight:700;}'
        '.sudoku-cell.filled{background:#f0fdf4;color:#166534;font-weight:600;}'
        '.sudoku-cell.empty{background:#fff;color:#475467;}'
        '</style>'
        f'<div class="sudoku-board" role="grid" aria-label="{n} by {n} Sudoku board">'
        + ''.join(cells) + '</div>'
    )


def clear_puzzle_results():
    """Discard any result from the previous puzzle when selection changes."""
    for key in ('solve_result', 'query_result', 'reasoning_trace'):
        st.session_state.pop(key, None)


def clear_solve_result():
    """An old solution does not apply to a newly selected algorithm."""
    st.session_state.pop('solve_result', None)


def clear_query_result():
    """An old verdict does not apply after the query inputs change."""
    st.session_state.pop('query_result', None)
    st.session_state.pop('reasoning_trace', None)


def trace_forward_chaining(kb):
    """Record actual rule firings in this KB for the tutor display.

    This is an app-specific explanation pass. The query verdict still comes
    from sudoku_solver.pl_bc_entails; no solver function is replaced here.
    """
    facts = {clause for clause in kb.clauses if is_prop_symbol(clause.op)}
    agenda = deque(clause for clause in kb.clauses if clause in facts)
    remaining = {
        clause: len(conjuncts(clause.args[0]))
        for clause in kb.clauses if clause.op == '==>'
    }
    proven = set()
    provenance = {}
    fired_count = 0

    while agenda:
        premise = agenda.popleft()
        if premise in proven:
            continue
        proven.add(premise)

        for rule in kb.clauses_with_premise(premise):
            remaining[rule] -= 1
            if remaining[rule] == 0:
                fired_count += 1
                conclusion = rule.args[1]
                if conclusion not in facts and conclusion not in provenance:
                    provenance[conclusion] = (rule, tuple(conjuncts(rule.args[0])))
                agenda.append(conclusion)

    return proven, facts, provenance, fired_count


def proof_steps(goal, facts, provenance):
    """Keep only the facts and fired rules that support a particular goal."""
    steps = []
    visited = set()

    def add_support(proposition):
        if proposition in visited:
            return
        visited.add(proposition)
        if proposition in facts:
            steps.append({'conclusion': proposition, 'premises': (), 'rule': None})
        else:
            rule, premises = provenance[proposition]
            for premise in premises:
                add_support(premise)
            steps.append({
                'conclusion': proposition, 'premises': premises, 'rule': rule,
            })

    add_support(goal)
    return steps


def atom_parts(proposition):
    """Read the assignment's Isr_c_v / Notr_c_v symbol names."""
    name = proposition.op
    prefix = 'Not' if name.startswith('Not') else 'Is'
    coordinates = name[len(prefix):]
    row, column, value = (int(part) for part in coordinates.split('_'))
    return prefix, row, column, value


def plain_fact(proposition):
    prefix, row, column, value = atom_parts(proposition)
    if prefix == 'Is':
        return f'Row {row}, column {column} contains {value}.'
    return f'Row {row}, column {column} cannot contain {value}.'


def step_explanation(step, box_h, box_w):
    """Turn one recorded fact or rule firing into a plain-English sentence."""
    conclusion = step['conclusion']
    premises = step['premises']
    prefix, row, column, value = atom_parts(conclusion)

    if step['rule'] is None:
        return f'Given: row {row}, column {column} already contains {value}.'

    if prefix == 'Not' and len(premises) == 1:
        _, source_row, source_column, _ = atom_parts(premises[0])
        if source_row == row:
            location = f'row {row} already contains {value} in column {source_column}'
        elif source_column == column:
            location = f'column {column} already contains {value} in row {source_row}'
        else:
            location = (
                f'the same {box_h} × {box_w} box already contains {value} '
                f'at row {source_row}, column {source_column}'
            )
        return f'Eliminate {value} from row {row}, column {column}: {location}.'

    eliminated = ', '.join(str(atom_parts(premise)[3]) for premise in premises)
    return (
        f'Last candidate for row {row}, column {column}: '
        f'{eliminated} have all been eliminated, so the value is {value}.'
    )


n, box_h, box_w = pool['n'], pool['box_h'], pool['box_w']

# --- 1. Puzzle selection & visual board display ---
# TODO: a dropdown/selectbox to pick a puzzle by index from pool['puzzles'].
# TODO: render the grid (e.g. a table or grid of st.columns), showing given
# cells and empty cells differently (e.g. bold givens, blank otherwise).
# Implemented below; the starter requirements remain for reference.
puzzle_index = st.selectbox(
    'Choose a puzzle',
    options=range(len(pool['puzzles'])),
    format_func=lambda i: f'Puzzle {i + 1} · {pool["puzzles"][i]["given_count"]} givens',
    key='puzzle_index',
    on_change=clear_puzzle_results,
)
puzzle = pool['puzzles'][puzzle_index]
givens = parse_givens(puzzle)
st.caption(f'{n} × {n} grid · {box_h} × {box_w} boxes · {len(givens)} given cells')
st.markdown(board_html(n, box_h, box_w, givens), unsafe_allow_html=True)

# --- 2. Full-grid auto-solver, with algorithm selection ---
# TODO: a radio/selectbox letting the user choose forward chaining
# (solve_full_grid_fc) or backward chaining (solve_full_grid_bc).
# TODO: a button that times and calls the chosen solver on
# (n, box_h, box_w, givens), then displays the solved grid and the elapsed
# time.
st.subheader('Solve the full grid')
algorithm = st.radio(
    'Inference algorithm',
    options=('Forward chaining', 'Backward chaining'),
    horizontal=True,
    key='solve_algorithm',
    on_change=clear_solve_result,
)

if st.button('Solve puzzle', key='solve_button', type='primary'):
    clear_solve_result()
    solver = solve_full_grid_fc if algorithm == 'Forward chaining' else solve_full_grid_bc
    with st.spinner(f'Solving with {algorithm.lower()}...'):
        started = time.perf_counter()
        solved = solver(n, box_h, box_w, givens)
        elapsed = time.perf_counter() - started

    expected_cells = {(r, c) for r in range(1, n + 1) for c in range(1, n + 1)}
    if set(solved) != expected_cells or any(solved.get(cell) != value for cell, value in givens.items()):
        st.error('The solver did not return a complete grid consistent with the givens.')
    else:
        st.session_state['solve_result'] = {
            'puzzle_index': puzzle_index,
            'algorithm': algorithm,
            'grid': solved,
            'seconds': elapsed,
        }

result = st.session_state.get('solve_result')
if result and result['puzzle_index'] == puzzle_index and result['algorithm'] == algorithm:
    st.success(f'{algorithm} solved the grid in {result["seconds"]:.2f} seconds.')
    st.markdown(board_html(n, box_h, box_w, givens, result['grid']), unsafe_allow_html=True)

# --- 3. Targeted cell entailment query ---
# TODO: number inputs for row (r), column (c), value (v).
# TODO: a button that builds the definite KB, calls
# pl_bc_entails(kb, atom('Is', r, c, v)), and displays True/False.
st.subheader('Check a cell/value')
row_col, column_col, value_col = st.columns(3)
with row_col:
    query_row = st.number_input(
        'Row (r)', min_value=1, max_value=n, value=1, step=1,
        key='query_row', on_change=clear_query_result,
    )
with column_col:
    query_column = st.number_input(
        'Column (c)', min_value=1, max_value=n, value=1, step=1,
        key='query_column', on_change=clear_query_result,
    )
with value_col:
    query_value = st.number_input(
        'Value (v)', min_value=1, max_value=n, value=1, step=1,
        key='query_value', on_change=clear_query_result,
    )

if st.button('Check entailment', key='query_button'):
    with st.spinner('Checking the definite knowledge base...'):
        query_kb = build_definite_kb(n, box_h, box_w, givens)
        verdict = pl_bc_entails(
            query_kb, atom('Is', query_row, query_column, query_value)
        )
        proven, facts, provenance, fired_count = trace_forward_chaining(query_kb)
        positive_goal = atom('Is', query_row, query_column, query_value)
        excluded_goal = atom('Not', query_row, query_column, query_value)
        trace_agrees = (positive_goal in proven) == verdict
        proof_goal = positive_goal if verdict else (
            excluded_goal if excluded_goal in proven else None
        )
        recorded_steps = (
            proof_steps(proof_goal, facts, provenance)
            if trace_agrees and proof_goal is not None else []
        )
    st.session_state['query_result'] = {
        'puzzle_index': puzzle_index,
        'row': query_row,
        'column': query_column,
        'value': query_value,
        'verdict': verdict,
    }
    st.session_state['reasoning_trace'] = {
        'puzzle_index': puzzle_index,
        'row': query_row,
        'column': query_column,
        'value': query_value,
        'consistent': trace_agrees,
        'proof_goal': proof_goal,
        'steps': recorded_steps,
        'fired_count': fired_count,
    }

query_result = st.session_state.get('query_result')
if query_result and (
    query_result['puzzle_index'] == puzzle_index
    and query_result['row'] == query_row
    and query_result['column'] == query_column
    and query_result['value'] == query_value
):
    st.info(
        f'Is({query_row}, {query_column}, {query_value}) entailed? '
        f'**{query_result["verdict"]}**'
    )

# --- 4. Reasoning trace ("tutor mode") ---
# TODO: instrument your forward- or backward-chaining approach to record each
# reasoning step (which rule fired, on what premises, producing what
# conclusion) as it answers the query above.
# TODO: render that trace as human-readable output -- e.g. a sequence of
# st.expander(...) blocks, one per step, each with a plain-English sentence
# -- not a raw list/dict dump.
#
# Keep the core solver functions in sudoku_solver.py; do not duplicate them here.
trace = st.session_state.get('reasoning_trace')
if trace and query_result and (
    trace['puzzle_index'] == puzzle_index
    and trace['row'] == query_row
    and trace['column'] == query_column
    and trace['value'] == query_value
):
    st.subheader('Reasoning trace')
    if not trace['consistent']:
        st.error('The recorded forward proof disagrees with the backward-chaining verdict.')
    elif trace['proof_goal'] is None:
        st.warning('No proof of this value or its elimination was derived from the current KB.')
    else:
        st.caption(
            'Verdict: backward chaining. Explanation: recorded forward chaining '
            'over the same definite knowledge base.'
        )
        if not query_result['verdict']:
            st.caption(
                'The queried value was not entailed. The steps below show how the '
                'knowledge base eliminated that candidate.'
            )
        given_count = sum(step['rule'] is None for step in trace['steps'])
        deduction_count = len(trace['steps']) - given_count
        given_word = 'given' if given_count == 1 else 'givens'
        deduction_word = 'deduction' if deduction_count == 1 else 'deductions'
        st.caption(
            f'{given_count} supporting {given_word} and '
            f'{deduction_count} rule {deduction_word} '
            f'from {trace["fired_count"]} total rule firings in this puzzle.'
        )
        for number, step in enumerate(trace['steps'], start=1):
            explanation = step_explanation(step, box_h, box_w)
            with st.expander(
                f'Step {number}: {explanation}',
                expanded=(number == len(trace['steps'])),
            ):
                if step['rule'] is None:
                    st.write('This fact is one of the puzzle givens.')
                else:
                    st.write('Premises used by this rule:')
                    for premise in step['premises']:
                        st.write(f'• {plain_fact(premise)}')
                    st.write(f'Conclusion: {plain_fact(step["conclusion"])}')
