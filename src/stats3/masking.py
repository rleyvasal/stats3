"""R-style shorthand inside stats3 calls, for notebooks.

Two rewrites, both limited to calls of stats3's own functions:

* Formulas: ``t_test(body_mass_g ~ sex)`` is not Python (``~`` has no
  binary form), so before the cell is parsed the formula argument is quoted:
  ``t_test("body_mass_g ~ sex")``.
* Bare column names: ``binom_test(sex == "male")`` and
  ``cor_test(bill_length_mm, bill_depth_mm, by=species)``. After parsing,
  a name that the notebook has not defined becomes ``pl.col("name")``. A
  name the notebook defines keeps its value, as anywhere in Python.

``stats3.jupyter`` installs both in IPython. When tidy3 is loaded its own
rewriting may get to the names first; either way they become columns.
"""

from __future__ import annotations

import ast
import builtins
import io
import tokenize
from typing import Any, Iterable

COL_NAME = "__stats3_col__"

# Every stats3 function whose arguments name columns.
FUNCTIONS = frozenset({
    "bayes_lm", "bayes_glm", "polr",
    "aov", "bartlett_test", "binom_test", "chisq_test", "cohens_d", "cor_test",
    "eta_squared", "fisher_test", "fligner_test", "friedman_test", "glm", "glm_nb", "hedges_g",
    "kruskal_test", "ks_test", "levene_test", "lm", "mantelhaen_test", "mcnemar_test",
    "oneway_test", "p_adjust", "pairwise_prop_test", "pairwise_t_test",
    "pairwise_wilcox_test", "poisson_test", "prop_test", "rlm", "rq", "shapiro_test", "t_test",
    "tukey_hsd", "var_test", "vcov_cl", "vcov_nw", "wilcox_test",
})


# formulas ------------------------------------------------------------------------


def quote_formulas(source: str) -> str:
    """Quote ``y ~ g`` arguments of stats3 calls; leave everything else as is."""
    if "~" not in source:
        return source
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    edits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, tok in enumerate(tokens[:-1]):
        nxt = tokens[i + 1]
        if tok.type == tokenize.NAME and tok.string in FUNCTIONS and nxt.string == "(":
            if i > 0 and tokens[i - 1].string == ".":
                prev = tokens[i - 2] if i > 1 else None
                if prev is None or prev.string != "stats3":
                    continue
            edits.extend(_formula_arguments(tokens, i + 1))
    if not edits:
        return source
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def at(pos: tuple[int, int]) -> int:
        return offsets[pos[0] - 1] + pos[1]

    out = source
    for start, end in sorted(edits, key=lambda e: at(e[0]), reverse=True):
        text = " ".join(source[at(start): at(end)].split())
        out = out[: at(start)] + repr(text) + out[at(end):]
    return out


def _formula_arguments(tokens: list[tokenize.TokenInfo], open_index: int):
    """Spans of top-level arguments after ``(`` that contain a bare ``~``."""
    depth = 0
    arg: list[tokenize.TokenInfo] = []
    for tok in tokens[open_index + 1:]:
        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            continue
        if tok.string in "([{" and tok.type == tokenize.OP:
            depth += 1
        elif tok.string in ")]}" and tok.type == tokenize.OP:
            if depth == 0:
                yield from _span(arg)
                return
            depth -= 1
        elif tok.string == "," and depth == 0:
            yield from _span(arg)
            arg = []
            continue
        arg.append(tok)


def _span(arg: list[tokenize.TokenInfo]):
    if not arg:
        return
    is_keyword = len(arg) > 1 and arg[0].type == tokenize.NAME and arg[1].string == "="
    if is_keyword or not any(t.string == "~" and t.type == tokenize.OP for t in arg):
        return
    # A unary ~ at the start (``~mask``) is Python; a formula has a left side.
    if arg[0].string == "~":
        return
    yield arg[0].start, arg[-1].end


def stats3_formula_transform(lines: list[str]) -> list[str]:
    """IPython input transformer: quote formulas in stats3 calls."""
    source = "".join(lines)
    out = quote_formulas(source)
    return out.splitlines(keepends=True) if out != source else lines


stats3_formula_transform.stats3_transform = True  # type: ignore[attr-defined]


# bare names ------------------------------------------------------------------------


def _assigned_names(tree: ast.AST) -> set[str]:
    """Names the cell itself binds (they are values, not columns)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _is_stats3_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in FUNCTIONS
    return (
        isinstance(func, ast.Attribute)
        and func.attr in FUNCTIONS
        and isinstance(func.value, ast.Name)
        and func.value.id == "stats3"
    )


class Stats3MaskTransformer(ast.NodeTransformer):
    """Turn undefined bare names inside stats3 calls into ``pl.col(name)``."""

    stats3_transform = True

    def __init__(self, namespace: dict[str, Any] | None = None):
        self.namespace = namespace if namespace is not None else {}
        self._local: set[str] = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self._local = _assigned_names(node)
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if _is_stats3_call(node):
            node.args = [self._mask(arg) for arg in node.args]
            for keyword in node.keywords:
                keyword.value = self._mask(keyword.value)
        return node

    def _defined(self, name: str) -> bool:
        return name in self.namespace or name in self._local or hasattr(builtins, name)

    def _mask(self, expr: ast.expr) -> ast.expr:
        transformer = self

        class Names(ast.NodeTransformer):
            def visit_Name(self, name: ast.Name) -> ast.AST:
                if isinstance(name.ctx, ast.Load) and not transformer._defined(name.id):
                    call = ast.Call(
                        func=ast.Name(id=COL_NAME, ctx=ast.Load()),
                        args=[ast.Constant(value=name.id)],
                        keywords=[],
                    )
                    return ast.copy_location(call, name)
                return name

            def visit_Call(self, call: ast.Call) -> ast.AST:
                # Leave other functions' names (np.log, ...) alone; their
                # arguments are still masked.
                call.args = [self.visit(a) for a in call.args]
                for keyword in call.keywords:
                    keyword.value = self.visit(keyword.value)
                return call

            def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
                return node

            def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                return node

        out = Names().visit(expr)
        return ast.fix_missing_locations(out)


def is_stats3_transform(obj: Any) -> bool:
    return bool(getattr(obj, "stats3_transform", False))


def mask_source(source: str, namespace: Iterable[str] | dict[str, Any] = ()) -> str:
    """Both rewrites on a piece of source, for tests and other front ends."""
    namespace = namespace if isinstance(namespace, dict) else {name: None for name in namespace}
    tree = ast.parse(quote_formulas(source))
    tree = Stats3MaskTransformer(namespace).visit(tree)
    return ast.unparse(ast.fix_missing_locations(tree))
