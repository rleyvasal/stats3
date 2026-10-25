"""Nonlinear least squares, following R's nls.

The model is an arithmetic expression over columns and named parameters,
``"y ~ a * exp(-b * x)"``, read by a small parser of its own (it is not a
model formula). Gradients come from symbolic differentiation of the parsed
expression, where R takes forward differences, so the fit follows R's
iterations with exact derivatives, as R does for its self-starting models.
"""

from __future__ import annotations

import math
import re

import numpy as np
import polars as pl

from stats3._data import Data, expression, step
from stats3.core import _check_conf_level, _group_label, _sorted_keys, _special, _warn
from stats3.models import NA_ACTIONS, Models, _Fit, _Frame, _ModelStep, _table_text


# expressions -------------------------------------------------------------------------------
#
# Nodes are tuples: ("num", value), ("var", name), ("neg", a), (op, a, b) for
# + - * / ^, and ("call", function, (arguments...)).

_TOKEN = re.compile(r"""
    \s*(?:
      (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?)
    | (?P<name>(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*|`[^`]+`)
    | (?P<op>\*\*|[-+*/^(),=])
    )""", re.VERBOSE)

_CONSTANTS = {"pi": math.pi, "Inf": math.inf, "TRUE": 1.0, "FALSE": 0.0}

# R's self-starting models: their arguments, and the model over them.
_SELF_START = {
    "SSlogis": (("input", "Asym", "xmid", "scal"), "Asym/(1+exp((xmid-input)/scal))"),
    "SSasymp": (("input", "Asym", "R0", "lrc"), "Asym+(R0-Asym)*exp(-exp(lrc)*input)"),
    "SSmicmen": (("input", "Vm", "K"), "Vm*input/(K+input)"),
    "SSgompertz": (("x", "Asym", "b2", "b3"), "Asym*exp(-b2*b3^x)"),
    "SSfpl": (("input", "A", "B", "xmid", "scal"), "A+(B-A)/(1+exp((xmid-input)/scal))"),
    "SSweibull": (("x", "Asym", "Drop", "lrc", "pwr"), "Asym-Drop*exp(-exp(lrc)*x^pwr)"),
}


def _num(v):
    return ("num", float(v))


def _is(node, value):
    return node[0] == "num" and node[1] == value


def _fold(op, a, b):
    with np.errstate(all="ignore"):
        x, y = np.float64(a[1]), np.float64(b[1])
        return _num({"+": x + y, "-": x - y, "*": x * y, "/": x / y, "^": x ** y}[op])


def _add(a, b):
    if _is(a, 0):
        return b
    if _is(b, 0):
        return a
    return _fold("+", a, b) if a[0] == b[0] == "num" else ("+", a, b)


def _sub(a, b):
    if _is(b, 0):
        return a
    if _is(a, 0):
        return _neg(b)
    return _fold("-", a, b) if a[0] == b[0] == "num" else ("-", a, b)


def _mul(a, b):
    if _is(a, 0) or _is(b, 0):
        return _num(0)
    if _is(a, 1):
        return b
    if _is(b, 1):
        return a
    return _fold("*", a, b) if a[0] == b[0] == "num" else ("*", a, b)


def _div(a, b):
    if _is(a, 0):
        return _num(0)
    if _is(b, 1):
        return a
    return _fold("/", a, b) if a[0] == b[0] == "num" else ("/", a, b)


def _neg(a):
    if a[0] == "num":
        return _num(-a[1])
    return a[1] if a[0] == "neg" else ("neg", a)


def _pow(a, b):
    if _is(b, 1):
        return a
    if _is(b, 0):
        return _num(1)
    return _fold("^", a, b) if a[0] == b[0] == "num" else ("^", a, b)


def _call(name, *args):
    return ("call", name, tuple(args))


def _sinpi(x):
    r = np.fmod(x, 2.0)
    out = np.sin(np.pi * r)
    return np.where(np.fmod(r, 1.0) == 0, 0.0, out)


def _cospi(x):
    r = np.fmod(np.abs(x), 2.0)
    out = np.cos(np.pi * r)
    return np.where(r == 0.5, 0.0, np.where(r == 1.5, 0.0, out))


def _dnorm(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _functions():
    s = _special()
    one = _num(1)
    return {
        "exp": (np.exp, lambda u: _call("exp", u)),
        "log": (np.log, lambda u: _div(one, u)),
        "log2": (np.log2, lambda u: _div(one, _mul(u, _num(math.log(2))))),
        "log10": (np.log10, lambda u: _div(one, _mul(u, _num(math.log(10))))),
        "log1p": (np.log1p, lambda u: _div(one, _add(one, u))),
        "expm1": (np.expm1, lambda u: _call("exp", u)),
        "sqrt": (np.sqrt, lambda u: _div(one, _mul(_num(2), _call("sqrt", u)))),
        "abs": (np.abs, lambda u: _call("sign", u)),
        "sign": (np.sign, lambda u: _num(0)),
        "sin": (np.sin, lambda u: _call("cos", u)),
        "cos": (np.cos, lambda u: _neg(_call("sin", u))),
        "tan": (np.tan, lambda u: _div(one, _pow(_call("cos", u), _num(2)))),
        "sinpi": (_sinpi, lambda u: _mul(_num(math.pi), _call("cospi", u))),
        "cospi": (_cospi, lambda u: _neg(_mul(_num(math.pi), _call("sinpi", u)))),
        "tanpi": (lambda x: _sinpi(x) / _cospi(x),
                  lambda u: _div(_num(math.pi), _pow(_call("cospi", u), _num(2)))),
        "asin": (np.arcsin, lambda u: _div(one, _call("sqrt", _sub(one, _pow(u, _num(2)))))),
        "acos": (np.arccos, lambda u: _neg(_div(one, _call("sqrt", _sub(one, _pow(u, _num(2))))))),
        "atan": (np.arctan, lambda u: _div(one, _add(one, _pow(u, _num(2))))),
        "sinh": (np.sinh, lambda u: _call("cosh", u)),
        "cosh": (np.cosh, lambda u: _call("sinh", u)),
        "tanh": (np.tanh, lambda u: _div(one, _pow(_call("cosh", u), _num(2)))),
        "asinh": (np.arcsinh, lambda u: _div(one, _call("sqrt", _add(_pow(u, _num(2)), one)))),
        "acosh": (np.arccosh, lambda u: _div(one, _call("sqrt", _sub(_pow(u, _num(2)), one)))),
        "atanh": (np.arctanh, lambda u: _div(one, _sub(one, _pow(u, _num(2))))),
        "gamma": (s.gamma, lambda u: _mul(_call("gamma", u), _call("digamma", u))),
        "lgamma": (s.gammaln, lambda u: _call("digamma", u)),
        "digamma": (s.psi, lambda u: _call("trigamma", u)),
        "trigamma": (lambda x: s.polygamma(1, x), lambda u: _call("psigamma", u, _num(2))),
        "factorial": (lambda x: s.gamma(x + 1),
                      lambda u: _mul(_call("gamma", _add(u, one)), _call("digamma", _add(u, one)))),
        "lfactorial": (lambda x: s.gammaln(x + 1), lambda u: _call("digamma", _add(u, one))),
        "pnorm": (s.ndtr, lambda u: _call("dnorm", u)),
        "dnorm": (_dnorm, lambda u: _neg(_mul(u, _call("dnorm", u)))),
        "plogis": (s.expit, lambda u: _call("dlogis", u)),
        "dlogis": (lambda x: s.expit(x) * s.expit(-x),
                   lambda u: _mul(_call("dlogis", u), _sub(one, _mul(_num(2), _call("plogis", u))))),
    }


_FUNCTIONS: dict | None = None


def _function_table():
    global _FUNCTIONS
    if _FUNCTIONS is None:
        _FUNCTIONS = _functions()
    return _FUNCTIONS


class _Parser:
    def __init__(self, text, verb):
        self.text, self.verb, self.tokens, pos = text, verb, [], 0
        while pos < len(text):
            if text[pos:].strip() == "":
                break
            match = _TOKEN.match(text, pos)
            if match is None or match.end() == pos:
                raise ValueError(f"{verb}(): cannot read the model at {text[pos:].strip()!r}")
            kind = match.lastgroup
            value = match.group(kind)
            if kind == "num":
                value = float(value.rstrip("L"))
            elif kind == "name" and value.startswith("`"):
                value = value[1:-1]
            elif kind == "op" and value == "**":
                value = "^"
            self.tokens.append((kind, value))
            pos = match.end()
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self, value=None):
        token = self.peek()
        if token[0] is None or (value is not None and token[1] != value):
            want = repr(value) if value else "more"
            raise ValueError(f"{self.verb}(): expected {want} in the model {self.text.strip()!r}")
        self.i += 1
        return token

    def done(self):
        if self.i != len(self.tokens):
            raise ValueError(f"{self.verb}(): unexpected {self.peek()[1]!r} in the model {self.text.strip()!r}")

    def sum(self):
        node = self.product()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = (op, node, self.product())
        return node

    def product(self):
        node = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            node = (op, node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            return ("^", node, self.unary())  # right associative, and 2^-1 is 0.5
        return node

    def atom(self):
        kind, value = self.take()
        if kind == "num":
            return _num(value)
        if kind == "op" and value == "(":
            node = self.sum()
            self.take(")")
            return node
        if kind == "name":
            if self.peek() == ("op", "("):
                return self.call(value)
            return ("var", value)
        raise ValueError(f"{self.verb}(): unexpected {value!r} in the model {self.text.strip()!r}")

    def arguments(self):
        self.take("(")
        args = []
        while self.peek() != ("op", ")"):
            name = None
            if self.peek()[0] == "name" and self.i + 1 < len(self.tokens) and self.tokens[self.i + 1] == ("op", "="):
                name = self.take()[1]
                self.take("=")
            args.append((name, self.sum()))
            if self.peek() != ("op", ")"):
                self.take(",")
        self.take(")")
        return args

    def call(self, name):
        args = self.arguments()
        if name == "cbind":
            raise ValueError(f"{self.verb}(): cbind() is only for the whole right-hand side, with algorithm='plinear'")
        if name in _SELF_START:
            return _expand_self_start(name, args, self.verb)[0]
        if name == "I" and len(args) == 1:
            return args[0][1]
        if any(n is not None for n, _ in args):
            raise ValueError(f"{self.verb}(): named arguments are not supported in {name}()")
        values = [a for _, a in args]
        if name == "log" and len(values) == 2:
            return _div(_call("log", values[0]), _call("log", values[1]))
        if name == "psigamma" and len(values) in (1, 2):
            order = values[1] if len(values) == 2 else _num(0)
            if order[0] != "num":
                raise ValueError(f"{self.verb}(): psigamma()'s deriv must be a number")
            return _call("psigamma", values[0], order)
        if name not in _function_table():
            raise ValueError(f"{self.verb}(): unknown function {name}() in the model")
        if len(values) != 1:
            raise ValueError(f"{self.verb}(): {name}() takes one argument")
        return _call(name, values[0])


def _substitute(node, mapping):
    kind = node[0]
    if kind == "var":
        return mapping.get(node[1], node)
    if kind == "num":
        return node
    if kind == "neg":
        return ("neg", _substitute(node[1], mapping))
    if kind == "call":
        return ("call", node[1], tuple(_substitute(a, mapping) for a in node[2]))
    return (kind, _substitute(node[1], mapping), _substitute(node[2], mapping))


def _expand_self_start(name, args, verb):
    formals, body = _SELF_START[name]
    matched, positional = {}, []
    for arg_name, value in args:
        if arg_name is None:
            positional.append(value)
        elif arg_name in formals and arg_name not in matched:
            matched[arg_name] = value
        else:
            raise ValueError(f"{verb}(): {name}() has no argument {arg_name!r}")
    free = [f for f in formals if f not in matched]
    if len(positional) != len(free):
        raise ValueError(f"{verb}(): {name}() takes the arguments {', '.join(formals)}")
    matched.update(zip(free, positional))
    model = _Parser(body, verb).sum()
    return _substitute(model, matched), (name, matched)


def _names(node, out):
    kind = node[0]
    if kind == "var":
        out.setdefault(node[1], None)
    elif kind == "neg":
        _names(node[1], out)
    elif kind == "call":
        for a in node[2]:
            _names(a, out)
    elif kind != "num":
        _names(node[1], out)
        _names(node[2], out)
    return out


def _depends(node, name):
    kind = node[0]
    if kind == "var":
        return node[1] == name
    if kind == "num":
        return False
    if kind == "neg":
        return _depends(node[1], name)
    if kind == "call":
        return any(_depends(a, name) for a in node[2])
    return _depends(node[1], name) or _depends(node[2], name)


def _derivative(node, name):
    """The symbolic derivative of ``node`` in the parameter ``name``."""
    if not _depends(node, name):
        return _num(0)
    kind = node[0]
    if kind == "var":
        return _num(1)
    if kind == "neg":
        return _neg(_derivative(node[1], name))
    if kind == "call":
        fn, args = node[1], node[2]
        if fn == "psigamma":
            outer = _call("psigamma", args[0], _num(args[1][1] + 1))
        else:
            outer = _function_table()[fn][1](args[0])
        return _mul(outer, _derivative(args[0], name))
    a, b = node[1], node[2]
    da, db = _derivative(a, name), _derivative(b, name)
    if kind == "+":
        return _add(da, db)
    if kind == "-":
        return _sub(da, db)
    if kind == "*":
        return _add(_mul(da, b), _mul(a, db))
    if kind == "/":
        return _sub(_div(da, b), _div(_mul(a, db), _pow(b, _num(2))))
    # a^b: the log term only where the exponent depends on the parameter.
    term = _mul(_mul(b, _pow(a, _sub(b, _num(1)))), da) if _depends(a, name) else _num(0)
    if _depends(b, name):
        term = _add(term, _mul(_mul(_pow(a, b), _call("log", a)), db))
    return term


def _evaluate(node, env, cache):
    hit = cache.get(node)
    if hit is not None:
        return hit
    kind = node[0]
    if kind == "num":
        value = node[1]
    elif kind == "var":
        value = env[node[1]]
    elif kind == "neg":
        value = -_evaluate(node[1], env, cache)
    elif kind == "call":
        args = [_evaluate(a, env, cache) for a in node[2]]
        if node[1] == "psigamma":
            order = int(node[2][1][1])
            value = _special().psi(args[0]) if order == 0 else _special().polygamma(order, args[0])
        else:
            value = _function_table()[node[1]][0](np.asarray(args[0], dtype=float))
    else:
        a, b = _evaluate(node[1], env, cache), _evaluate(node[2], env, cache)
        value = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power}[kind](
            np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    cache[node] = value
    return value


def _split_formula(text, verb):
    depth, quoted = 0, False
    for i, c in enumerate(text):
        quoted ^= c == "`"
        if quoted:
            continue
        depth += (c == "(") - (c == ")")
        if c == "~" and depth == 0:
            return text[:i], text[i + 1:]
    raise ValueError(f"{verb}(): the model needs a '~', as in 'y ~ a * exp(-b * x)'")


class _Expression:
    """A parsed nls model: response, right-hand side column(s), the names used."""

    def __init__(self, text, verb):
        if not isinstance(text, str):
            raise TypeError(f"{verb}(): the model must be a string, such as 'y ~ a * exp(-b * x)'")
        self.text = " ".join(text.split())
        left, right = _split_formula(text, verb)
        if left.strip():
            parser = _Parser(left, verb)
            self.lhs = parser.sum()
            parser.done()
            self.response = left.strip()
        else:
            self.lhs, self.response = _num(0), None
        parser = _Parser(right, verb)
        self.self_start = None
        self.matrix = False
        token = parser.peek()
        nxt = parser.tokens[1] if len(parser.tokens) > 1 else (None, None)
        if token == ("name", "cbind") and nxt == ("op", "("):
            parser.take()
            args = parser.arguments()
            if not args:
                raise ValueError(f"{verb}(): cbind() needs at least one column")
            self.columns = [a for _, a in args]
            self.column_names = [n for n, _ in args]
            self.matrix = True
        elif token[0] == "name" and token[1] in _SELF_START and nxt == ("op", "("):
            name = parser.take()[1]
            node, self.self_start = _expand_self_start(name, parser.arguments(), verb)
            self.columns, self.column_names = [node], [None]
        else:
            self.columns, self.column_names = [parser.sum()], [None]
        parser.done()
        if self.self_start is not None and parser.i != len(parser.tokens):
            self.self_start = None
        names: dict[str, None] = {}
        for node in [self.lhs, *self.columns]:
            _names(node, names)
        self.names = list(names)


# the iterations -----------------------------------------------------------------------------


_NOT_FINITE = "Missing value or an infinity produced when evaluating the model"


class _Model:
    """An expression bound to its columns: values and gradients in the parameters."""

    def __init__(self, expr, params, env, n, verb):
        self.expr, self.params, self.env, self.n, self.verb = expr, list(params), env, n, verb
        self.gradients = [[_derivative(c, p) for p in self.params] for c in expr.columns]
        with np.errstate(all="ignore"):
            lhs = np.broadcast_to(np.asarray(_evaluate(expr.lhs, env, {}), dtype=float), (n,)).copy()
        if not np.all(np.isfinite(lhs)):
            raise ValueError(f"{verb}(): the response has missing or infinite values")
        self.lhs = lhs

    def evaluate(self, theta, columns=None):
        """The right-hand side columns (n, k) and their gradient (n, k, p)."""
        env = dict(self.env if columns is None else columns)
        env.update(zip(self.params, (float(t) for t in theta)))
        n = self.n if columns is None else _rows(columns)
        cache: dict = {}
        with np.errstate(all="ignore"):
            values = np.column_stack([np.broadcast_to(np.asarray(_evaluate(c, env, cache), dtype=float), (n,))
                                      for c in self.expr.columns])
            if columns is not None:
                return values, None
            grad = np.empty((n, len(self.expr.columns), len(self.params)))
            for k, row in enumerate(self.gradients):
                for j, node in enumerate(row):
                    grad[:, k, j] = np.broadcast_to(np.asarray(_evaluate(node, env, cache), dtype=float), (n,))
        if not (np.all(np.isfinite(values)) and np.all(np.isfinite(grad))):
            raise ValueError(f"{self.verb}(): {_NOT_FINITE}")
        return values, grad



def _rows(columns):
    for value in columns.values():
        if np.ndim(value):
            return len(value)
    return 1


def _rank_ok(x, tol=1e-7):
    """Full column rank by the rule of R's dqrdc2 (as models._qr)."""
    if x.shape[1] == 0:
        return True
    if x.shape[0] < x.shape[1]:
        return False
    r = np.linalg.qr(x, mode="r")
    norms = np.linalg.norm(x, axis=0)
    return bool(np.all(norms > 0) and np.all(np.abs(np.diag(r)) >= tol * norms))


class _GaussNewton:
    """R's nlsModel: the Gauss-Newton increment and relative-offset criterion."""

    def __init__(self, model, sw, scale_offset):
        self.model, self.sw = model, sw
        self.npar = len(model.params)
        n = model.n
        self.scale_offset = (n - self.npar) * scale_offset ** 2 if scale_offset else 0.0

    def set(self, pars):
        self.pars = np.array(pars, dtype=float)
        values, grad = self.model.evaluate(self.pars)
        self.rhs = values[:, 0]
        self.resid = self.sw * (self.model.lhs - self.rhs)
        self.dev = float(np.sum(self.resid ** 2))
        self.gradient = self.sw[:, None] * grad[:, 0, :]
        singular = not _rank_ok(self.gradient)
        if not singular:
            self.q, self.r = np.linalg.qr(self.gradient)
        return singular

    def conv(self):
        if self.npar == 0:
            return 0.0
        top = self.q.T @ self.resid
        rest = self.resid - self.q @ top
        return math.sqrt(float(top @ top) / (self.scale_offset + float(rest @ rest)))

    def incr(self):
        from scipy.linalg import solve_triangular

        return solve_triangular(self.r, self.q.T @ self.resid)

    def fitted(self):
        return self.rhs

    def coefficients(self):
        return self.pars

    def design(self):
        return self.gradient


class _Plinear:
    """R's nlsModel.plinear: Golub-Pereyra steps with the linear terms projected out.

    The Householder factors and full Q'v come from LAPACK, whose reflectors
    are LINPACK's (x + sign(x1)|x|e1): R's criterion reads components of
    Q'v beyond the column space when there are more linear terms than
    nonlinear ones."""

    def __init__(self, model, sw, scale_offset):
        self.model, self.sw = model, sw
        self.p1, self.p2 = len(model.expr.columns), len(model.params)
        n = model.n
        self.fac = (n - self.p1 - self.p2) / (self.p1 + self.p2)
        self.scale_offset = (n - self.p1) * scale_offset ** 2 if scale_offset else 0.0
        self.y = sw * model.lhs

    def set(self, pars):
        self.pars = np.array(pars, dtype=float)
        self.rhs, self.grad = self.model.evaluate(self.pars)
        x = self.sw[:, None] * self.rhs
        self.factor, self.tau = _householder(x)
        qty = _apply_qt(self.factor, self.tau, self.y)
        rest = np.r_[np.zeros(self.p1), qty[self.p1:]]
        self.resid = _apply_q(self.factor, self.tau, rest)
        self.dev = float(np.sum(self.resid ** 2))
        if not _rank_ok(x):
            return True
        from scipy.linalg import solve_triangular

        self.lin = solve_triangular(np.triu(self.factor[:self.p1, :self.p1]), qty[:self.p1])
        self.cc = rest
        return False

    def _ddot(self):
        return np.einsum("nkj,k->nj", self.grad, self.lin)

    def conv(self):
        from scipy.linalg import solve_triangular

        b = _apply_qt(self.factor, self.tau, self.sw[:, None] * self._ddot())
        top = np.einsum("nkj,n->kj", self.sw[:, None, None] * self.grad, self.resid)
        r = np.triu(self.factor[:self.p1, :self.p1])
        b[:self.p1] = solve_triangular(r.T, top, lower=True)
        self.b = b
        factor, tau = _householder(b)
        rr = _apply_qt(factor, tau, self.cc)
        head = float(np.sum(rr[:self.p1] ** 2))
        tail = float(np.sum(rr[self.p1:] ** 2))
        return math.sqrt(self.fac * head / (self.scale_offset + tail))

    def incr(self):
        if not _rank_ok(self.b):
            raise ValueError("singular matrix 'a' in solve")
        return np.linalg.lstsq(self.b, self.cc, rcond=None)[0]

    def fitted(self):
        return self.rhs @ self.lin

    def coefficients(self):
        return np.r_[self.pars, self.lin]

    def design(self):
        return self.sw[:, None] * np.column_stack([self._ddot(), self.rhs])


def _householder(x):
    from scipy.linalg import lapack

    factor, tau, _, info = lapack.dgeqrf(np.asfortranarray(x, dtype=float))
    return factor, tau


def _apply_qt(factor, tau, c, trans="T"):
    from scipy.linalg import lapack

    c = np.asarray(c, dtype=float)
    vector = c.ndim == 1
    c2 = np.asfortranarray(c[:, None] if vector else c)
    k = min(factor.shape)
    if k == 0:
        return c.copy()
    lwork = max(1, c2.shape[1]) * 64
    out, _, info = lapack.dormqr("L", trans, factor[:, :k], tau[:k], c2, lwork)
    return out[:, 0] if vector else out


def _apply_q(factor, tau, c):
    return _apply_qt(factor, tau, c, "N")


class _Stop(Exception):
    pass


def _iterate(state, start, control, verb):
    """R's nls_iter: Gauss-Newton increments with step halving."""
    if state.set(start):
        raise ValueError(f"{verb}(): singular gradient matrix at initial parameter estimates")
    pars, dev, fac = state.pars.copy(), state.dev, 1.0
    maxiter, tol, min_factor = control["maxiter"], control["tol"], control["min_factor"]
    conv, i = -1.0, 0

    def finish(code, message):
        if not control["warn_only"]:
            raise ValueError(f"{verb}(): {message}")
        _warn(f"{verb}(): {message}")
        return {"is_conv": False, "fin_iter": i, "fin_tol": conv, "stop_code": code, "message": message}

    while i < maxiter:
        conv = state.conv()
        if conv <= tol:
            return {"is_conv": True, "fin_iter": i, "fin_tol": conv, "stop_code": 0, "message": "converged"}
        increment = state.incr()
        while fac >= min_factor:
            trial = pars + fac * increment
            if state.set(trial):
                return finish(1, "singular gradient")
            if state.dev <= dev:
                dev, pars, fac = state.dev, trial, min(2 * fac, 1.0)
                break
            fac /= 2
        if fac < min_factor:
            return finish(2, f"step factor {fac:g} reduced below 'minFactor' of {min_factor:g}")
        i += 1
    return finish(3, f"number of iterations exceeded maximum of {maxiter}")


def _port(state, start, lower, upper, control, verb):
    """Bounded least squares for algorithm='port'.

    R calls the PORT library's nl2sol; this runs a trust-region reflective
    solver with the same analytic gradient to the bounded optimum, so the
    estimates agree with R's to its convergence accuracy, not its path."""
    from scipy.optimize import least_squares

    if np.any(start < lower) or np.any(start > upper):
        return _port_failure("initial par violates constraints", control, verb, state, start)
    if state.set(start):
        raise ValueError(f"{verb}(): singular gradient matrix at initial parameter estimates")

    def residuals(theta):
        state.set(theta)
        return state.resid

    def jacobian(theta):
        state.set(theta)
        return -state.gradient

    result = least_squares(residuals, start, jac=jacobian, bounds=(lower, upper), method="trf",
                           xtol=1e-15, ftol=1e-15, gtol=1e-15, x_scale="jac",
                           max_nfev=max(control["maxiter"], 1) * 4)
    state.set(np.clip(result.x, lower, upper))
    if result.status <= 0:
        return _port_failure("iteration limit reached without convergence (10)", control, verb, state, None)
    return {"is_conv": True, "fin_iter": int(result.nfev), "fin_tol": None, "stop_code": 0,
            "message": "relative convergence (4)"}


def _port_failure(message, control, verb, state, start):
    if not control["warn_only"]:
        raise ValueError(f"{verb}(): Convergence failure: {message}")
    _warn(f"{verb}(): Convergence failure: {message}")
    if start is not None:
        state.set(start)
    return {"is_conv": False, "fin_iter": 0, "fin_tol": None, "stop_code": 1, "message": message}


_CONTROL = {"maxiter": 50, "tol": 1e-5, "min_factor": 1 / 1024, "warn_only": False, "scale_offset": 0.0}


def _control(control, verb):
    out = dict(_CONTROL)
    if control is None:
        return out
    if not isinstance(control, dict):
        raise TypeError(f"{verb}(): control must be a dict, such as {{'maxiter': 100, 'tol': 1e-6}}")
    aliases = {"minFactor": "min_factor", "warnOnly": "warn_only", "scaleOffset": "scale_offset"}
    for key, value in control.items():
        key = aliases.get(key, key)
        if key not in out:
            raise ValueError(f"{verb}(): unknown control setting {key!r}; use {', '.join(_CONTROL)}")
        out[key] = value
    maxiter = out["maxiter"]
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)) or maxiter < 0:
        raise ValueError(f"{verb}(): control maxiter must be a non-negative integer")
    for key in ("tol", "min_factor"):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{verb}(): control {key} must be positive")
    if not isinstance(out["scale_offset"], (int, float)) or out["scale_offset"] < 0:
        raise ValueError(f"{verb}(): control scale_offset must be non-negative")
    return out


def _fit_core(expr, params, start, columns, n, weights, algorithm, control, lower, upper, verb):
    model = _Model(expr, params, columns, n, verb)
    sw = np.sqrt(weights)
    if algorithm == "plinear":
        state = _Plinear(model, sw, control["scale_offset"])
    else:
        if expr.matrix:
            raise ValueError(f"{verb}(): cbind() on the right-hand side needs algorithm='plinear'")
        state = _GaussNewton(model, sw, control["scale_offset"])
    start = np.asarray(start, dtype=float)
    if algorithm == "port":
        info = _port(state, start, lower, upper, control, verb)
    else:
        info = _iterate(state, start, control, verb)
    return model, state, info


# self-starting models -----------------------------------------------------------------------


def _sorted_xy(x, y):
    """R's sortedXyData: the mean response at each distinct input, by input."""
    keys = np.array([float(f"{v:.15g}") for v in np.asarray(x, dtype=float)])
    xs = np.unique(keys)
    ys = np.array([np.mean(y[keys == v]) for v in xs])
    keep = ~(np.isnan(xs) | np.isnan(ys))
    return xs[keep], ys[keep]


def _lm(y, *xs):
    x = np.column_stack([np.ones(len(y)), *xs])
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _right_asymptote(y):
    lo, hi = np.min(y), np.max(y)
    last = np.abs(np.array([lo, hi]) - y[-1])
    return hi + (hi - lo) / 8 if np.argmin(last) == 1 else lo - (hi - lo) / 8


def _left_asymptote(y):
    lo, hi = np.min(y), np.max(y)
    first = np.abs(np.array([lo, hi]) - y[0])
    return hi + (hi - lo) / 8 if np.argmin(first) == 1 else lo - (hi - lo) / 8


def _plinear_coef(text, x, y, start, control, verb):
    # R fits these with numericDeriv; exact gradients here move the starting
    # values by about 1e-8 relative, as R's forward differences round.
    expr = _Expression(text, verb)
    _, state, _ = _fit_core(expr, list(start), list(start.values()), {"x": x, "y": y}, len(x),
                            np.ones(len(x)), "plinear", control, None, None, verb)
    return state.coefficients()


def _asymptotic(x, y, control, verb):
    """R's NLSstAsymptotic: b0 + b1*(1 - exp(-exp(lrc) * x))."""
    rt = _right_asymptote(y)
    lrc = math.log(-_lm(np.log(np.abs(y - rt)), x)[1])
    lrc, b0, b1 = _plinear_coef("y ~ cbind(1, 1 - exp(-exp(lrc) * x))", x, y, {"lrc": lrc}, control, verb)
    return b0, b1, lrc


def _initial(kind, x, y, control, verb):
    """Starting values from R's self-start initial functions, in model order."""
    x, y = _sorted_xy(x, y)
    n = len(x)
    with np.errstate(all="ignore"):
        if kind == "SSlogis":
            if n < 4:
                raise ValueError(f"{verb}(): too few distinct input values to fit a logistic model")
            lo, dz = np.min(y), np.ptp(y)
            z = (y - lo + .05 * dz) / (1.1 * dz)
            aux = _lm(x, np.log(z / (1 - z)))
            xmid, scal, lin = _plinear_coef("y ~ 1/(1 + exp((xmid - x)/scal))", x, y,
                                            {"xmid": aux[0], "scal": aux[1]}, control, verb)
            return [lin, xmid, scal]
        if kind == "SSasymp":
            if n < 3:
                raise ValueError(f"{verb}(): too few distinct input values to fit an asymptotic regression model")
            if n > 3:
                lrc = math.log(-_lm(np.log(np.abs(y - _right_asymptote(y))), x)[1])
                lrc, asym, r0 = _plinear_coef("y ~ cbind(1 - exp(-exp(lrc) * x), exp(-exp(lrc) * x))",
                                              x, y, {"lrc": lrc}, control, verb)
                return [asym, r0, lrc]
            ydiff = np.diff(y)
            if np.prod(ydiff) <= 0:
                raise ValueError(f"{verb}(): cannot fit an asymptotic regression model to these data")
            frac = (y[2] - y[0]) / (y[1] - y[0])
            xdiff = np.diff(x)
            if xdiff[0] != xdiff[1]:
                raise ValueError(f"{verb}(): too few observations to fit an asymptotic regression model")
            expm_rd = frac - 1
            rc = -math.log(expm_rd) / xdiff[0]
            expm_rx1 = math.exp(-rc * x[0])
            bma = ydiff[0] / (expm_rx1 * (expm_rd - 1))
            asym = y[0] - bma * expm_rx1
            return [asym, bma + asym, math.log(rc)]
        if kind == "SSmicmen":
            if n < 3:
                raise ValueError(f"{verb}(): too few distinct input values to fit a Michaelis-Menten model")
            b = _lm(1 / y, 1 / x)
            k, vm = _plinear_coef("y ~ x/(K + x)", x, y, {"K": abs(b[1] / b[0])}, control, verb)
            return [vm, k]
        if kind == "SSgompertz":
            if n < 4:
                raise ValueError(f"{verb}(): too few distinct input values to fit the Gompertz model")
            _, b1, lrc = _asymptotic(x, np.log(np.abs(y)), control, verb)
            b2, b3, lin = _plinear_coef("y ~ exp(-b2 * b3^x)", x, y,
                                        {"b2": b1, "b3": math.exp(-math.exp(lrc))}, control, verb)
            return [lin, b2, b3]
        if kind == "SSfpl":
            if n < 5:
                raise ValueError(f"{verb}(): too few distinct input values to fit a four-parameter logistic")
            lo, drng = np.min(y), np.ptp(y)
            prop = (y - lo + .05 * drng) / (1.1 * drng)
            ir = _lm(x, np.log(prop / (1 - prop)))
            xmid, lscal, a, ba = _plinear_coef("y ~ cbind(1, 1/(1 + exp((xmid - x)/exp(lscal))))", x, y,
                                               {"xmid": ir[0], "lscal": math.log(abs(ir[1]))}, control, verb)
            return [a, a + ba, xmid, math.exp(lscal)]
        if kind == "SSweibull":
            if n < 5:
                raise ValueError(f"{verb}(): too few distinct input values to fit the Weibull growth model")
            if np.any(x < 0):
                raise ValueError(f"{verb}(): all 'x' values must be non-negative to fit the Weibull growth model")
            rasym, lasym = _right_asymptote(y), _left_asymptote(y)
            used = x > 0
            b = _lm(np.log(-np.log((rasym - y[used]) / (rasym - lasym))), np.log(x[used]))
            lrc, pwr, asym, drop = _plinear_coef("y ~ cbind(1, -exp(-exp(lrc) * x^pwr))", x, y,
                                                 {"lrc": b[0], "pwr": b[1]}, control, verb)
            return [asym, drop, lrc, pwr]
    raise AssertionError(kind)


# model frames -------------------------------------------------------------------------------


def _frames(data, columns, weights, na_action, verb):
    groups = list(data._groups)
    for g in groups:
        if g in columns:
            raise ValueError(f"{verb}(): {g!r} is a grouping column; it cannot be in the model")
    exprs = {}
    schema = data._lf.collect_schema()
    for c in columns:
        dtype = schema[c]
        if not (dtype.is_numeric() or dtype == pl.Boolean):
            raise ValueError(f"{verb}(): column {c!r} is not numeric")
        exprs[f"__src_{c}"] = pl.col(c).cast(pl.Float64)
    if weights is not None:
        exprs["__weights"] = expression(weights).cast(pl.Float64)
    frame = data._lf.select(pl.int_range(pl.len(), dtype=pl.Int64).alias("__row"),
                            *[pl.col(g) for g in groups], **exprs).collect()
    keep = pl.lit(True)
    for name in exprs:
        keep = keep & pl.col(name).is_not_null() & pl.col(name).is_not_nan()
    full = frame if na_action == "exclude" else None
    height = frame.height
    frame = frame.filter(keep)
    if na_action == "fail" and frame.height < height:
        raise ValueError(f"{verb}(): missing values in the model's variables (na_action='fail')")
    if groups:
        keys = _sorted_keys(data, frame.select(groups).unique(maintain_order=True), groups)
        parts = frame.partition_by(groups, as_dict=True, maintain_order=True)
        pieces = [(dict(zip(groups, key)), parts[tuple(key)]) for key in keys.iter_rows()]
    else:
        pieces = [({}, frame)]
    out = []
    source = [*groups, *[pl.col(f"__src_{c}").alias(c) for c in columns]]
    for key, part in pieces:
        n = part.height
        if n == 0:
            raise ValueError(f"{verb}(): 0 (non-NA) cases{_group_label(key, groups)}")
        w = part["__weights"].to_numpy() if weights is not None else np.ones(n)
        if np.any(w < 0):
            raise ValueError(f"{verb}(): missing or negative weights not allowed{_group_label(key, groups)}")
        values = {c: part[f"__src_{c}"].to_numpy() for c in columns}
        all_rows = all_index = None
        if full is not None:
            whole = full
            for g in groups:
                whole = whole.filter(pl.col(g).is_null() if key[g] is None else pl.col(g) == key[g])
            all_rows, all_index = whole.select(source), whole["__row"].to_numpy()
        out.append(_Frame(key, part.select(source), np.zeros(0), values, w, np.zeros(n), None,
                          part["__row"].to_numpy(), all_rows, all_index))
    return out


# the fit ------------------------------------------------------------------------------------


class NonlinearModel(_Fit):
    """An nls fit: parameter estimates, their covariance, and the fitted curve."""

    kind = "nls"

    def __init__(self, expr, frame, data, start, algorithm, control, lower, upper, verb="nls"):
        super().__init__(expr, None, frame, data, verb)
        self.expr, self.algorithm, self.control = expr, algorithm, control
        columns = dict(frame.variables)
        self.columns = columns
        self.n = len(frame.weights)
        self.prior_weights = frame.weights
        params, start = self._start(expr, columns, start, algorithm, control, verb)
        p = len(params)
        lower = _bounds(lower, -np.inf, params, verb)
        upper = _bounds(upper, np.inf, params, verb)
        self.lower, self.upper = lower, upper
        env = {**_CONSTANTS, **columns}
        self.env = env
        self.model, self.state, self.info = _fit_core(expr, params, start, env, self.n, frame.weights,
                                                      algorithm, control, lower, upper, verb)
        self.params = params
        if algorithm == "plinear":
            k = len(expr.columns)
            if k == 1:
                lin = [".lin"]
            elif all(name is not None for name in expr.column_names):
                lin = [f".lin.{name}" for name in expr.column_names]
            else:
                lin = [f".lin{j + 1}" for j in range(k)]
            self.parameter_names = [*params, *lin]
        else:
            self.parameter_names = list(params)
        self.coef = self.state.coefficients()
        self.converged = self.info["is_conv"]
        self.fitted = self.state.fitted()
        self.resid = self.state.resid
        self.deviance = self.state.dev
        self.rank = len(self.coef)
        w = frame.weights
        self.nobs = int(np.sum(w != 0))
        self.df_residual = self.nobs - self.rank
        rdf = int(np.sum(w > 0)) - self.rank
        self.sigma = math.sqrt(self.deviance / rdf) if rdf > 0 else math.nan
        self._rdf = rdf
        design = self.state.design()
        r = np.linalg.qr(design, mode="r")
        from scipy.linalg import solve_triangular

        inverse = solve_triangular(r, np.eye(r.shape[0]))
        self.cov_unscaled = inverse @ inverse.T
        self._cov = self.cov_unscaled * self.sigma ** 2
        self._p = p

    @staticmethod
    def _start(expr, columns, start, algorithm, control, verb):
        names = [n for n in expr.names if n not in _CONSTANTS or n in columns]
        if start is None:
            if expr.self_start is not None:
                kind, args = expr.self_start
                formals = _SELF_START[kind][0]
                params = []
                for formal in formals[1:]:
                    node = args[formal]
                    if node[0] != "var":
                        raise ValueError(f"{verb}(): {kind}()'s {formal} must be a parameter name to self-start")
                    params.append(node[1])
                missing = [c for c in _names(args[formals[0]], {}) if c not in columns and c not in _CONSTANTS]
                if missing:
                    raise KeyError(f"{verb}(): column {missing[0]!r} does not exist")
                env = {**_CONSTANTS, **columns}
                with np.errstate(all="ignore"):
                    x = np.broadcast_to(np.asarray(_evaluate(args[formals[0]], env, {}), dtype=float),
                                        (len(next(iter(columns.values()))) if columns else 1,))
                    y = np.broadcast_to(np.asarray(_evaluate(expr.lhs, env, {}), dtype=float), x.shape)
                values = _initial(kind, x, y, control, verb)
                return params, values
            unknown = [n for n in names if n not in columns]
            if algorithm == "plinear":
                raise ValueError(f"{verb}(): no starting values specified")
            if not unknown:
                raise ValueError(f"{verb}(): no parameters to fit")
            _warn(f"{verb}(): No starting values specified for some parameters.\nInitializing "
                  + ", ".join(f"'{u}'" for u in unknown)
                  + " to '1.'.\nConsider specifying 'start' or using a selfStart model")
            return unknown, [1.0] * len(unknown)
        if not isinstance(start, dict) or not start:
            raise TypeError(f"{verb}(): start must be a dict of parameter names and values")
        params, values = [], []
        for name, value in start.items():
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{verb}(): start value for {name!r} must be one number")
            params.append(str(name))
            values.append(float(value))
        unknown = [n for n in names if n not in columns and n not in params]
        if unknown:
            raise ValueError(f"{verb}(): parameters without starting value in 'data': {', '.join(unknown)}")
        return params, values

    @property
    def coefficients(self):
        return dict(zip(self.parameter_names, self.coef.tolist()))

    # tables ---------------------------------------------------------------------------------

    def _check_vcov(self, vcov):
        if vcov is not None:
            raise ValueError("nls(): vcov= is for lm/glm; this model supplies its own covariance")

    def tidy(self, conf_int=False, conf_level=.95, vcov=None):
        s = _special()
        self._check_vcov(vcov)
        se = np.sqrt(np.diag(self._cov))
        stat = self.coef / se
        p = 2 * s.stdtr(self._rdf, -np.abs(stat)) if self._rdf > 0 else np.full(self.rank, np.nan)
        cols = {"term": self.parameter_names, "estimate": self.coef, "std_error": se,
                "statistic": stat, "p_value": p}
        if conf_int:
            _check_conf_level(conf_level)
            low, high = _profile_intervals(self, conf_level)
            cols.update(conf_low=low, conf_high=high)
        frame = pl.DataFrame(cols)
        return self._with_key(frame.with_columns(pl.col(c).fill_nan(None) for c in frame.columns if c != "term"))

    def vcov_table(self, vcov=None):
        self._check_vcov(vcov)
        return self._with_key(pl.DataFrame({"term": self.parameter_names,
            **{name: self._cov[:, j] for j, name in enumerate(self.parameter_names)}}))

    def log_lik(self):
        w = self.prior_weights
        zero = w == 0
        n = int(np.sum(~zero))
        return -n * (math.log(2 * math.pi) + 1 - math.log(n) - float(np.sum(np.log(w + zero))) / n
                     + math.log(float(np.sum(self.resid ** 2)))) / 2

    def glance(self):
        ll = self.log_lik()
        k = self.rank + 1
        return self._with_key(pl.DataFrame({
            "sigma": [self.sigma], "is_conv": [self.converged], "fin_tol": [self.info["fin_tol"]],
            "logLik": [ll], "AIC": [-2 * ll + 2 * k], "BIC": [-2 * ll + math.log(self.nobs) * k],
            "deviance": [self.deviance], "df_residual": [self.df_residual], "nobs": [self.nobs]},
            schema_overrides={"fin_tol": pl.Float64}))

    def _predict_new(self, newdata):
        data = Data.read(newdata, None, "predict")
        needed = [n for n in _rhs_names(self.expr) if n not in self.params and (n not in _CONSTANTS or n in self.columns)]
        missing = [n for n in needed if n not in data.columns]
        if missing:
            raise KeyError(f"predict(): column {missing[0]!r} does not exist in newdata")
        frame = data._lf.select(*[pl.col(c).cast(pl.Float64) for c in needed]).collect() if needed \
            else pl.DataFrame({"__n": np.zeros(data._lf.select(pl.len()).collect().item())})
        columns = {**_CONSTANTS, **{c: frame[c].to_numpy() for c in needed}}
        if not needed:
            columns["__n"] = frame["__n"].to_numpy()
        pars = self.coef[:self._p]
        values, _ = self.model.evaluate(pars, columns)
        fit = values @ self.coef[self._p:] if self.algorithm == "plinear" else values[:, 0]
        return fit, data

    def predict(self, newdata=None):
        if newdata is None:
            return self._data.write(self._padded(pl.DataFrame({"fit": self.fitted})))
        fit, data = self._predict_new(newdata)
        return data.write(pl.DataFrame({"fit": fit}))

    def augment(self, newdata=None):
        if newdata is None:
            cols = pl.DataFrame({".fitted": self.fitted, ".resid": self.model.lhs - self.fitted})
            return pl.concat([self._rows(), self._padded(cols)], how="horizontal")
        fit, data = self._predict_new(newdata)
        return data.write(pl.concat([data._lf.collect(), pl.DataFrame({".fitted": fit})], how="horizontal"))

    def anova_table(self, test=None):
        raise TypeError("anova(): 'anova' is only defined for sequences of nls fits; compare two or more")

    def anova_type(self, kind, test=None):
        return self.anova_table(test)

    def __repr__(self):
        text = (f"nls({self.expr.text}, algorithm={self.algorithm!r}){_key_line(self.key)}\n{_table_text(self.tidy())}"
                f"\nResidual standard error: {self.sigma:.4g} on {self._rdf} degrees of freedom")
        if not self.converged:
            text += f"\nNot converged: {self.info['message']}"
        return text


def _key_line(key):
    return "" if not key else "   [" + ", ".join(f"{k} = {v!r}" for k, v in key.items()) + "]"


def _rhs_names(expr):
    names: dict[str, None] = {}
    for node in expr.columns:
        _names(node, names)
    return list(names)


def _bounds(value, default, params, verb):
    p = len(params)
    if value is None:
        return np.full(p, default)
    if isinstance(value, dict):
        unknown = set(value) - set(params)
        if unknown:
            raise ValueError(f"{verb}(): bounds for unknown parameters {sorted(unknown)}")
        return np.array([float(value.get(name, default)) for name in params])
    values = np.atleast_1d(np.asarray(value, dtype=float))
    if len(values) == 0:
        return np.full(p, default)
    return np.resize(values, p)  # R's rep_len


# profile intervals --------------------------------------------------------------------------


def _profile_intervals(fit, level):
    """confint.nls's profile-t intervals, solved exactly.

    R's profile.nls steps along each parameter and interpolates a spline of
    the profile t statistic; here each end point is the root of the
    profile t statistic, with the other parameters refitted at each value.
    The cutoff is R's: qt on the number of rows less the nonlinear
    parameters (all rows of the fit, as length(fitted()) counts them)."""
    from scipy.optimize import brentq
    from scipy.stats import t as student

    s = _special()
    rows = len(fit.frame.all_index) if fit.frame.all_index is not None else fit.n
    p = fit._p
    cutoff = float(student.ppf((1 + level) / 2, rows - p))
    limit = 10 * math.sqrt(float(s.fdtri(1, rows - p, 1 - (1 - level) / 4)))
    se = np.sqrt(np.diag(fit._cov))
    low, high = np.full(fit.rank, np.nan), np.full(fit.rank, np.nan)
    s2 = fit.sigma ** 2
    if not (s2 > 0) or p == 0:
        return low, high
    for j in range(p):
        for sign, out in ((-1, low), (1, high)):
            out[j] = _profile_end(fit, j, sign, cutoff, se[j], s2, limit, brentq)
    return low, high


def _profile_end(fit, j, sign, cutoff, se, s2, limit, brentq):
    base = fit.coef.copy()
    step = se * cutoff / 5
    previous, previous_pars = fit.coef[j], base
    for k in range(1, 1000):
        value = fit.coef[j] + sign * k * step
        if abs(value - fit.coef[j]) / se > limit:
            return math.nan
        if value < fit.lower[j] or value > fit.upper[j]:
            return math.nan
        dev, pars = _conditional(fit, j, value, previous_pars)
        if dev is None:
            return math.nan
        tau = math.sqrt(max(dev - fit.deviance, 0) / s2)
        if tau >= cutoff:
            start = previous_pars

            def gap(v):
                d, _ = _conditional(fit, j, v, start)
                if d is None:
                    raise _Stop
                return math.sqrt(max(d - fit.deviance, 0) / s2) - cutoff

            try:
                return brentq(gap, previous, value, xtol=1e-14 * max(abs(value), 1e-300), rtol=1e-15, maxiter=200)
            except _Stop:
                return math.nan
        previous, previous_pars = value, pars
    return math.nan


def _conditional(fit, j, value, start):
    """The least deviance with parameter j fixed at ``value``, and its parameters."""
    from scipy.optimize import least_squares

    model, sw, y = fit.model, np.sqrt(fit.prior_weights), fit.model.lhs
    p, plinear = fit._p, fit.algorithm == "plinear"
    free = [i for i in range(fit.rank) if i != j]
    lower = np.r_[fit.lower, np.full(fit.rank - p, -np.inf)][free]
    upper = np.r_[fit.upper, np.full(fit.rank - p, np.inf)][free]

    def full(theta):
        pars = np.empty(fit.rank)
        pars[j], pars[free] = value, theta
        return pars

    def parts(theta):
        pars = full(theta)
        values, grad = model.evaluate(pars[:p])
        if plinear:
            lin = pars[p:]
            fitted = values @ lin
            jac = np.column_stack([np.einsum("nkj,k->nj", grad, lin), values])
        else:
            fitted, jac = values[:, 0], grad[:, 0, :]
        return sw * (y - fitted), -sw[:, None] * jac[:, free]

    if not free:
        try:
            r, _ = parts(np.zeros(0))
        except ValueError:
            return None, None
        return float(r @ r), full(np.zeros(0))
    x0 = np.clip(np.asarray(start, dtype=float)[free], lower, upper)
    try:
        bounded = np.any(np.isfinite(lower)) or np.any(np.isfinite(upper))
        result = least_squares(lambda t: parts(t)[0], x0, jac=lambda t: parts(t)[1],
                               bounds=(lower, upper), method="trf" if bounded else "lm",
                               xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    except ValueError:
        return None, None
    if result.status <= 0:
        return None, None
    r = result.fun
    return float(r @ r), full(result.x)


# the verb -----------------------------------------------------------------------------------


@step
def nls(formula, *, start=None, algorithm="default", weights=None, na_action="omit",
        control=None, lower=None, upper=None):
    """Nonlinear least squares, as R's ``nls``.

    ``nls(data, "y ~ a * exp(-b * x)", start={"a": 1, "b": 0.1})``. The model
    is an arithmetic expression (``+ - * / ^``, R's math functions) of
    columns and parameters; names in ``start`` are parameters. Without
    ``start=``, R's self-starting models ``SSlogis``, ``SSasymp``,
    ``SSmicmen``, ``SSgompertz``, ``SSfpl`` and ``SSweibull`` find their own.

    ``algorithm`` is ``"default"`` (Gauss-Newton), ``"plinear"`` (Golub-Pereyra,
    with the conditionally linear terms as ``cbind(...)`` columns and
    coefficients ``.lin``) or ``"port"`` (with ``lower``/``upper`` bounds).
    ``control`` takes R's ``maxiter``, ``tol``, ``min_factor``, ``warn_only``
    and ``scale_offset``. ``weights=``, ``na_action`` and ``by=`` work as
    in ``lm``. ``tidy(conf_int=True)`` gives confint.nls's profile intervals.
    """
    if algorithm not in ("default", "plinear", "port"):
        raise ValueError("nls(): algorithm must be 'default', 'plinear' or 'port'")
    if na_action not in NA_ACTIONS:
        raise ValueError(f"nls(): na_action must be one of {list(NA_ACTIONS)}, got {na_action!r}")
    settings = _control(control, "nls")
    expr = _Expression(formula, "nls")
    if expr.matrix and algorithm != "plinear":
        raise ValueError("nls(): cbind() on the right-hand side needs algorithm='plinear'")
    if algorithm != "port" and (lower is not None or upper is not None):
        _warn('nls(): upper and lower bounds ignored unless algorithm = "port"')
        lower = upper = None
    params = set(start or {})

    def _apply(data):
        columns = [n for n in expr.names if n not in params and n in data.columns]
        unknown = [n for n in expr.names if n not in params and n not in data.columns and n not in _CONSTANTS]
        if start is not None and unknown:
            raise ValueError(f"nls(): parameters without starting value in 'data': {', '.join(unknown)}")
        frames = _frames(data, columns, weights, na_action, "nls")
        fits = [NonlinearModel(expr, frame, data, start, algorithm, settings, lower, upper)
                for frame in frames]
        return Models(fits, data) if data._groups else fits[0]

    return _ModelStep(_apply, "nls")


def _anova_nls(fits, test):
    """R's anovalist.nls: each model against the one before it."""
    s = _special()
    if test not in (None, "F"):
        raise ValueError("anova(): nls comparisons use the F test")
    responses = [f.expr.response for f in fits]
    if any(r != responses[0] for r in responses[1:]):
        raise ValueError("anova(): nls fits must share the response")
    df_r = [float(f.df_residual) for f in fits]
    ss_r = [f.deviance for f in fits]
    df, ss, stat, p = [None], [None], [None], [None]
    for i in range(1, len(fits)):
        d, q = df_r[i - 1] - df_r[i], ss_r[i - 1] - ss_r[i]
        f = pv = None
        if d > 0:
            f = (q / d) / (ss_r[i] / df_r[i])
            pv = float(s.fdtrc(d, df_r[i], f))
        elif d < 0:
            f = (q / d) / (ss_r[i - 1] / df_r[i - 1])
            pv = float(s.fdtrc(-d, df_r[i - 1], f))
        else:
            q = 0.0
        df.append(d)
        ss.append(q)
        stat.append(f)
        p.append(pv)
    return pl.DataFrame({"term": [f.expr.text for f in fits], "df_residual": df_r, "rss": ss_r,
                         "df": df, "sumsq": ss, "statistic": stat, "p_value": p},
                        schema_overrides={"df": pl.Float64, "sumsq": pl.Float64,
                                          "statistic": pl.Float64, "p_value": pl.Float64})
