"""Data in and out: pandas, Polars and tidy3 frames, without importing them.

Every test works on a ``Data``: a Polars LazyFrame, the grouping columns, and
the factor levels that order a grouping factor's levels. ``Data.read`` builds
one from whatever the caller passed, and ``Data.write`` turns the result back
into the same kind of frame:

========================  ==================  ===========================
input                     groups              factor levels
========================  ==================  ===========================
Polars DataFrame / Lazy   ``by=``             ``Enum`` columns
pandas DataFrame          ``by=``             ``category`` columns
tidy3 TidyFrame           its ``group_by``    its recorded levels
dict of columns           ``by=``             none
========================  ==================  ===========================

pandas and tidy3 are recognised by their class and module, so stats3 imports
neither; a TidyFrame can only exist when tidy3 is already loaded.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, Callable

import polars as pl

OUTPUTS = ("polars", "pandas", "tidy3")


def _is_pandas_frame(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "DataFrame" and cls.__module__.startswith("pandas")


def _is_pandas_series(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "Series" and cls.__module__.startswith("pandas")


def _is_tidy(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "TidyFrame" and cls.__module__.startswith("tidy3")


def is_frame(value: Any) -> bool:
    """True for anything stats3 can read as a table."""
    return (
        isinstance(value, (pl.DataFrame, pl.LazyFrame))
        or _is_pandas_frame(value)
        or _is_tidy(value)
        or (isinstance(value, Mapping) and not isinstance(value, str))
    )


def _tidy_expr(value: Any) -> bool:
    """A tidy3 expression (``col("x")``, ``sex == "male"``), duck-typed."""
    node = getattr(value, "node", None)
    return isinstance(node, tuple) and type(value).__module__.startswith("tidy3")


def column_name(spec: Any) -> str | None:
    """The column a spec names, or None when it is a computed expression.

    Accepts a string (tidy3's bare notebook names are strings too), a Polars
    column expression, or a tidy3 ``col("x")``.
    """
    if isinstance(spec, str):
        return str(spec)
    if isinstance(spec, pl.Expr):
        if spec.meta.is_column():
            return spec.meta.output_name()
        return None
    if _tidy_expr(spec) and spec.node[0] in {"col", "name"}:
        return str(spec.node[1])
    return None


def expression(spec: Any) -> pl.Expr:
    """A Polars expression for a column name, Polars expression, or tidy3
    expression (compiled with tidy3, which is loaded if one exists)."""
    if isinstance(spec, pl.Expr):
        return spec
    name = column_name(spec)
    if name is not None:
        return pl.col(name)
    if _tidy_expr(spec):
        return sys.modules["tidy3.expr"].to_polars(spec)
    raise TypeError(f"expected a column name or expression, got {spec!r}")


class Data:
    """One input table, read once: rows, groups, and factor levels."""

    def __init__(
        self,
        lf: pl.LazyFrame,
        groups: list[str],
        levels: dict[str, list[Any]],
        kind: str,
        source: Any = None,
    ):
        self._lf = lf
        self._groups = groups
        self._levels = levels
        self.kind = kind
        self.source = source
        self._names: list[str] | None = None

    @property
    def columns(self) -> list[str]:
        if self._names is None:
            self._names = list(self._lf.collect_schema().names())
        return self._names

    @classmethod
    def read(cls, data: Any, by: Any, verb: str) -> "Data":
        if isinstance(data, pl.LazyFrame):
            out = cls(data, [], {}, "lazy")
        elif isinstance(data, pl.DataFrame):
            out = cls(data.lazy(), [], {}, "polars")
        elif _is_pandas_frame(data):
            levels = {
                str(name): list(data[name].cat.categories)
                for name, dtype in data.dtypes.items()
                if str(dtype) == "category"
            }
            try:
                frame = pl.from_pandas(data)
            except ImportError as e:
                raise ImportError(
                    f"{verb}(): reading this pandas frame needs pyarrow: "
                    "pip install 'stats3[pandas]'"
                ) from e
            out = cls(frame.lazy(), [], levels, "pandas")
        elif _is_tidy(data):
            out = cls(data.lazy(), _tidy_groups(data), _tidy_levels(data), "tidy3", data)
        elif isinstance(data, Mapping):
            out = cls(pl.DataFrame(dict(data)).lazy(), [], {}, "polars")
        else:
            raise TypeError(
                f"{verb}(): expected a Polars or pandas DataFrame, a tidy3 frame, "
                f"or a dict of columns; got {type(data).__name__}"
            )
        if by is not None:
            if out._groups:
                raise ValueError(f"{verb}(): by= cannot be used on a grouped frame")
            specs = by if isinstance(by, (list, tuple)) else [by]
            out._groups = [out.column(spec, "by", verb) for spec in specs]
        return out

    def column(self, spec: Any, role: str, verb: str) -> str:
        name = column_name(spec)
        if name is None:
            raise ValueError(f"{verb}(): {role} must name one column, got {spec!r}")
        if name not in self.columns:
            shown = ", ".join(repr(c) for c in self.columns[:20])
            raise KeyError(f"{verb}(): column {name!r} does not exist. Columns: {shown}")
        return name

    def levels_of(self, name: str) -> list[Any] | None:
        """A column's factor levels in order, or None when it is not a factor."""
        if name in self._levels:
            return list(self._levels[name])
        dtype = self._lf.collect_schema()[name]
        return dtype.categories.to_list() if isinstance(dtype, pl.Enum) else None

    def write(self, result: pl.DataFrame, as_: str | None = None) -> Any:
        """The result as the kind of frame that came in (or ``as_``)."""
        kind = as_ or self.kind
        if kind == "lazy":
            return result.lazy()
        if kind == "polars":
            return result
        if kind == "pandas":
            return _to_pandas(result, {g: self._levels[g] for g in self._groups if g in self._levels})
        if kind == "tidy3":
            return _to_tidy(result, self)
        raise ValueError(f"as_ must be one of {OUTPUTS}, got {as_!r}")


def _tidy_groups(frame: Any) -> list[str]:
    tidy3 = sys.modules["tidy3"]
    groups = tidy3.group_vars(frame) if hasattr(tidy3, "group_vars") else frame._groups
    return [str(g) for g in (groups or [])]


def _tidy_levels(frame: Any) -> dict[str, list[Any]]:
    accessor = getattr(frame, "factor_levels", None)
    levels = accessor() if callable(accessor) else getattr(frame, "_category_levels", {})
    return {str(name): list(values) for name, values in (levels or {}).items()}


def _to_pandas(result: pl.DataFrame, levels: dict[str, list[Any]]) -> Any:
    try:
        frame = result.to_pandas()
    except ImportError as e:
        raise ImportError(
            "returning a pandas frame needs pandas and pyarrow: pip install 'stats3[pandas]'"
        ) from e
    for name, values in levels.items():
        frame[name] = frame[name].astype("category").cat.set_categories(values)
    return frame


def _to_tidy(result: pl.DataFrame, data: Data) -> Any:
    """A tidy3 frame that keeps the grouping columns' factor levels."""
    if "tidy3" not in sys.modules:
        raise ImportError("as_='tidy3' needs tidy3: pip install tidy3")
    levels = {g: data._levels[g] for g in data._groups if g in data._levels}
    if data.source is not None:
        try:
            return type(data.source)(result.lazy(), category_levels=levels)
        except TypeError:
            pass
    return sys.modules["tidy3"].tidy(result)


class Step:
    """A test waiting for data: ``data >> step``, or ``step(data)``.

    pandas and Polars frames have no ``>>`` of their own, and tidy3 frames
    hand ``>>`` to the right-hand side, so ``__rrshift__`` receives them all.
    """

    __slots__ = ("_fn", "name", "by", "as_")

    def __init__(self, fn: Callable[[Data], pl.DataFrame], name: str, by: Any = None, as_: str | None = None):
        self._fn = fn
        self.name = name
        self.by = by
        self.as_ = as_

    def __rrshift__(self, data: Any) -> Any:
        frame = Data.read(data, self.by, self.name)
        return frame.write(self._fn(frame), self.as_)

    def __call__(self, data: Any) -> Any:
        return self.__rrshift__(data)

    def __repr__(self) -> str:
        return f"<stats3 {self.name}: pipe data into it with >>>"


def step(fn: Callable[..., Step]) -> Callable[..., Any]:
    """Make a test usable as a pipe step or with the data first.

    ``t_test("y ~ g", by="g2")`` returns a ``Step``;
    ``t_test(frame, "y ~ g", by="g2")`` runs it on ``frame`` at once.
    Every test gets ``by=`` (dplyr's ``.by``) and ``as_=`` (the output frame
    type, by default the input's).
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args: Any, by: Any = None, as_: str | None = None, **kwargs: Any) -> Any:
        if as_ is not None and as_ not in OUTPUTS:
            raise ValueError(f"as_ must be one of {OUTPUTS}, got {as_!r}")
        if args and is_frame(args[0]) and not isinstance(args[0], str):
            data, rest = args[0], args[1:]
            made = fn(*rest, **kwargs)
            made.by, made.as_ = by, as_
            return made(data)
        made = fn(*args, **kwargs)
        made.by, made.as_ = by, as_
        return made

    return wrapper
