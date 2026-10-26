"""R model formulas and design matrices, as R's ``terms`` and ``model.matrix``.

A formula like ``body_mass_g ~ flipper_length_mm + species * sex`` becomes a
response, an intercept flag, offsets, and a list of terms; each term is a
tuple of variables (``("species", "sex")`` for ``species:sex``). Supported:

* ``+``, ``-``, ``*`` (main effects and interaction), ``:`` (interaction),
  ``(a + b)^2`` (interactions up to order 2), ``.`` (every other column),
  ``0`` / ``-1`` (no intercept), ``1``;
* variables that are columns or expressions of them: ``log(x)``, ``sqrt(x)``,
  ``exp(x)``, ``I(x^2)``, ``factor(cyl)``; backticks quote odd names;
* ``offset(...)`` terms, and ``cbind(successes, failures)`` responses for
  binomial models.

Design matrices follow R's rules: terms sorted by order, treatment contrasts
(the first level is the reference), full indicator coding when a factor's
margin is not in the model, and R's column names (``speciesChinstrap``,
``speciesChinstrap:sexmale``, ``dreamTRUE``).
"""

from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

_FUNCTIONS = {
    "log": lambda e: e.log(),
    "log2": lambda e: e.log(2),
    "log10": lambda e: e.log10(),
    "log1p": lambda e: e.log1p(),
    "exp": lambda e: e.exp(),
    "sqrt": lambda e: e.sqrt(),
    "abs": lambda e: e.abs(),
    "sin": lambda e: e.sin(),
    "cos": lambda e: e.cos(),
    "I": lambda e: e,
}
_FACTOR_CALLS = {"factor", "as.factor", "as_factor"}


@dataclass
class Formula:
    text: str
    response: str | None
    terms: list[tuple[str, ...]]
    intercept: bool
    offsets: list[str] = field(default_factory=list)
    # Contrasts per factor: "treatment" (R's default) or "sum" (contr.sum).
    contrasts: dict[str, str] = field(default_factory=dict)
    # The fit's offset= argument (a column or expression), kept so that
    # predictions on new rows add it, as R's predict does.
    offset_arg: Any = None

    @property
    def term_labels(self) -> list[str]:
        return [":".join(term) for term in self.terms]

    def variables(self) -> list[str]:
        seen: dict[str, None] = {}
        for term in self.terms:
            for var in term:
                seen.setdefault(var, None)
        return list(seen)


# parsing -------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    """Top-level formula tokens; a variable keeps its parentheses whole."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch in "+-*:^()":
            out.append(ch)
            i += 1
        elif ch == "`":
            j = text.index("`", i + 1)
            out.append(text[i: j + 1])
            i = j + 1
        else:
            j = i
            depth = 0
            while j < n:
                c = text[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0 and (c.isspace() or c in "+-*:^"):
                    break
                elif c in "\"'":
                    j = text.index(c, j + 1)
                j += 1
            out.append(text[i:j])
            i = j
    return out


class _Parser:
    """Recursive descent over R's formula operators: ``+ -`` < ``*`` < ``:`` < ``^``."""

    def __init__(self, tokens: list[str], verb: str):
        self.tokens = tokens
        self.pos = 0
        self.verb = verb
        self.order: dict[str, int] = {}
        self.offsets: list[str] = []

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"{self.verb}(): formula ends too early")
        self.pos += 1
        return tok

    # A value is (terms, intercept): terms an ordered list of frozensets.
    def parse(self) -> tuple[list[frozenset], bool | None]:
        value = self.sum()
        if self.peek() is not None:
            raise ValueError(f"{self.verb}(): unexpected {self.peek()!r} in formula")
        return value

    def sum(self):
        if self.peek() == "-":  # leading minus: ~ -1 + x
            self.take()
            terms, intercept = [], True
            terms, intercept = self._remove(terms, intercept, self.prod())
        else:
            terms, intercept = self.prod()
            intercept = True if intercept is None else intercept
        while self.peek() in ("+", "-"):
            op = self.take()
            right = self.prod()
            if op == "+":
                for t in right[0]:
                    if t not in terms:
                        terms.append(t)
                if right[1] is not None:
                    intercept = right[1]
            else:
                terms, intercept = self._remove(terms, intercept, right)
        return terms, intercept

    @staticmethod
    def _remove(terms, intercept, right):
        removed = set(right[0])
        terms = [t for t in terms if t not in removed]
        if right[1] is True:  # - 1
            intercept = False
        elif right[1] is False:  # - 0
            intercept = True
        return terms, intercept

    def prod(self):
        left = self.inter()
        while self.peek() == "*":
            self.take()
            right = self.inter()
            terms = list(left[0])
            for t in right[0] + _products(left[0], right[0]):
                if t not in terms:
                    terms.append(t)
            left = (terms, None)
        return left

    def inter(self):
        left = self.power()
        while self.peek() == ":":
            self.take()
            right = self.power()
            left = (_products(left[0], right[0]), None)
        return left

    def power(self):
        base = self.atom()
        if self.peek() == "^":
            self.take()
            k = int(self.take())
            terms = list(base[0])
            current = list(base[0])
            for _ in range(k - 1):
                current = _products(current, base[0])
                for t in current:
                    if t not in terms:
                        terms.append(t)
            base = (terms, None)
        return base

    def atom(self):
        tok = self.take()
        if tok == "(":
            value = self.sum()
            if self.take() != ")":
                raise ValueError(f"{self.verb}(): unbalanced parentheses in formula")
            return value[0], (value[1] if value[1] is False else None)
        if tok in ("1", "0"):
            return [], tok == "1"
        if tok.startswith("offset("):
            self.offsets.append(tok[len("offset("):-1].strip())
            return [], None
        name = tok
        self.order.setdefault(name, len(self.order))
        return [frozenset([name])], None


def _products(left: list[frozenset], right: list[frozenset]) -> list[frozenset]:
    out: list[frozenset] = []
    for a in left:
        for b in right:
            t = a | b
            if t not in out:
                out.append(t)
    return out


def parse_formula(text: str, verb: str, columns: list[str] | None = None) -> Formula:
    """Parse ``y ~ terms`` into a :class:`Formula`."""
    if not isinstance(text, str) or "~" not in text:
        raise ValueError(f"{verb}(): formula must look like 'y ~ x + ...', got {text!r}")
    lhs, _, rhs = text.partition("~")
    lhs, rhs = lhs.strip(), rhs.strip()
    if not rhs:
        raise ValueError(f"{verb}(): the formula has no right side: {text!r}")
    response = lhs or None
    if columns is not None and "." in _tokens(rhs):
        used = set(_column_names(response)) if response else set()
        expansion = " + ".join(_backtick(c) for c in columns if c not in used)
        rhs = " ".join(expansion if tok == "." else tok for tok in _tokens(rhs)) or "1"
    parser = _Parser(_tokens(rhs), verb)
    terms, intercept = parser.parse()
    order = parser.order
    ordered = [tuple(sorted(t, key=lambda v: order[v])) for t in terms]
    # R sorts terms by order (main effects, then two-way, ...), keeping
    # the formula's order within each.
    ordered.sort(key=len)
    return Formula(text=" ".join(text.split()), response=response, terms=ordered,
                   intercept=bool(intercept), offsets=parser.offsets)


def _backtick(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.]*", name) else f"`{name}`"


def _column_names(label: str) -> list[str]:
    """Column names a variable refers to."""
    names: list[str] = []
    tree = ast.parse(_python(label), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _FUNCTIONS and node.id not in {"factor", "as_factor", "cbind"}:
            names.append(_unmangle(node.id))
    return names


# evaluating variables -------------------------------------------------------------

_BACKTICK = re.compile(r"`([^`]*)`")


def _mangle(name: str) -> str:
    return "__bt_" + name.encode().hex() + "__"


def _unmangle(name: str) -> str:
    m = re.fullmatch(r"__bt_([0-9a-f]*)__", name)
    return bytes.fromhex(m.group(1)).decode() if m else name


def _python(label: str) -> str:
    """An R variable expression as Python source (``^`` -> ``**``,
    backticks and ``as.factor`` made into names)."""
    text = _BACKTICK.sub(lambda m: _mangle(m.group(1)), label)
    text = text.replace("as.factor(", "as_factor(").replace("^", "**")
    return text


@dataclass
class Variable:
    """One evaluated variable: numeric values, or factor codes and levels."""

    label: str
    values: np.ndarray | None = None
    codes: np.ndarray | None = None
    levels: list[Any] | None = None

    @property
    def is_factor(self) -> bool:
        return self.levels is not None


def variable_expr(label: str, verb: str) -> tuple[pl.Expr, bool]:
    """A Polars expression for a variable, and whether it is ``factor()``."""
    tree = ast.parse(_python(label), mode="eval").body
    as_factor = False
    if isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name) and tree.func.id in {"factor", "as_factor"}:
        if len(tree.args) != 1:
            raise ValueError(f"{verb}(): factor() takes one column in {label!r}")
        tree, as_factor = tree.args[0], True
    return _compile(tree, label, verb), as_factor


def _compile(node: ast.AST, label: str, verb: str) -> pl.Expr:
    if isinstance(node, ast.Name):
        return pl.col(_unmangle(node.id))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return pl.lit(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_compile(node.operand, label, verb)
    if isinstance(node, ast.BinOp):
        a, b = _compile(node.left, label, verb), _compile(node.right, label, verb)
        ops = {ast.Add: a.__add__, ast.Sub: a.__sub__, ast.Mult: a.__mul__,
               ast.Div: a.__truediv__, ast.Pow: a.__pow__}
        for kind, fn in ops.items():
            if isinstance(node.op, kind):
                return fn(b)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        if len(node.args) != 1 or node.keywords:
            raise ValueError(f"{verb}(): {node.func.id}() takes one argument in {label!r}")
        return _FUNCTIONS[node.func.id](_compile(node.args[0], label, verb).cast(pl.Float64))
    raise ValueError(
        f"{verb}(): cannot read {label!r}; use columns, arithmetic, "
        f"{', '.join(sorted(_FUNCTIONS))}, or factor()"
    )


def is_categorical(dtype: pl.DataType) -> bool:
    return dtype in (pl.String, pl.Categorical, pl.Boolean) or isinstance(dtype, (pl.Enum, pl.Categorical))


# design matrix -----------------------------------------------------------------------


@dataclass
class Design:
    """A model matrix with R's column names and term assignment."""

    matrix: np.ndarray
    names: list[str]
    assign: list[int]  # term index per column (0 = intercept)
    term_labels: list[str]
    levels: dict[str, list[Any]]  # factor levels per variable, for predict()
    factors: set[str]


def encode(frame: pl.DataFrame, label: str, as_factor: bool, levels: list[Any] | None) -> Variable:
    """Evaluate one variable on the model frame (column ``label`` holds it)."""
    series = frame[label]
    if as_factor or is_categorical(series.dtype):
        values = series.to_list()
        if levels is None:
            present = set(values)
            levels = sorted(present, key=lambda v: (v is None, v))
        else:
            present = set(values)
            levels = [lv for lv in levels if lv in present]
        index = {lv: i for i, lv in enumerate(levels)}
        try:
            codes = np.array([index[v] for v in values], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"factor {label!r} has a value not in its levels: {e.args[0]!r}") from None
        return Variable(label, codes=codes, levels=levels)
    return Variable(label, values=series.cast(pl.Float64).to_numpy())


def _level_name(level: Any) -> str:
    if isinstance(level, bool):
        return "TRUE" if level else "FALSE"
    if isinstance(level, float) and level.is_integer():
        return str(int(level))
    return str(level)


def build_design(
    formula: Formula, variables: dict[str, Variable], n: int, *, fixed_levels: dict[str, list[Any]] | None = None
) -> Design:
    """R's ``model.matrix`` for the formula's terms."""
    term_sets = [frozenset(t) for t in formula.terms]
    blocks: list[np.ndarray] = []
    names: list[str] = []
    assign: list[int] = []
    if formula.intercept:
        blocks.append(np.ones((n, 1)))
        names.append("(Intercept)")
        assign.append(0)
    for index, term in enumerate(formula.terms, start=1):
        pieces: list[tuple[list[str], list[np.ndarray]]] = []
        for var in term:
            v = variables[var]
            if not v.is_factor:
                pieces.append(([var], [v.values]))
                continue
            margin = frozenset(term) - {var}
            # R: contrasts when the term without this factor is in the model
            # (the intercept stands for the empty term), else indicators.
            contrasts = (margin in term_sets) if margin else formula.intercept
            kind = formula.contrasts.get(var, "treatment")
            if not contrasts:
                cols = [(v.codes == i).astype(float) for i in range(len(v.levels))]
                pieces.append(([f"{var}{_level_name(lv)}" for lv in v.levels], cols))
            elif kind == "sum":
                # contr.sum: level j against the last level, named var1, var2, ...
                last = len(v.levels) - 1
                cols = [(v.codes == j).astype(float) - (v.codes == last).astype(float) for j in range(last)]
                pieces.append(([f"{var}{j + 1}" for j in range(last)], cols))
            else:
                cols = [(v.codes == i).astype(float) for i in range(1, len(v.levels))]
                pieces.append(([f"{var}{_level_name(lv)}" for lv in v.levels[1:]], cols))
        # The first variable varies fastest, as in R.
        for combo in itertools.product(*[range(len(p[0])) for p in reversed(pieces)]):
            combo = combo[::-1]
            col = None
            parts = []
            for (labels, cols), k in zip(pieces, combo):
                col = cols[k] if col is None else col * cols[k]  # 1 * x is x: a lone column is itself
                parts.append(labels[k])
            blocks.append(col[:, None])
            names.append(":".join(parts))
            assign.append(index)
    matrix = np.hstack(blocks) if blocks else np.empty((n, 0))
    levels = {name: v.levels for name, v in variables.items() if v.is_factor}
    return Design(matrix, names, assign, formula.term_labels, levels,
                  {name for name, v in variables.items() if v.is_factor})
