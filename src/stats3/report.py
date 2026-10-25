"""Side-by-side tables of fitted models, in the spirit of R's modelsummary:
each model a column, its coefficients with standard errors (or intervals)
below them, then fit statistics.

``modelsummary`` returns a frame (with ``part``, ``term`` and ``statistic``
columns, as ``modelsummary(output = "data.frame")``), or the table as
Markdown or HTML text.
"""

from __future__ import annotations

import html
import math
from typing import Any, Mapping

import polars as pl

from stats3._data import OUTPUTS, Data

STATISTICS = ("std_error", "conf_int", "statistic", "p_value", None)
GOF_LABELS = {
    "nobs": "Num.Obs.",
    "r_squared": "R2",
    "adj_r_squared": "R2 Adj.",
    "AIC": "AIC",
    "BIC": "BIC",
    "logLik": "Log.Lik.",
    "sigma": "Sigma",
    "deviance": "Deviance",
    "statistic": "F",
}
DEFAULT_GOF = ("nobs", "r_squared", "adj_r_squared", "AIC", "BIC", "logLik")
STARS = ((0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, "+"))


def _named(models: tuple[Any, ...]) -> list[tuple[str, Any]]:
    """(column name, fit) pairs: a dict's keys, a grouped fit's groups, or (1), (2), ..."""
    from stats3.models import Models, _is_model

    if len(models) == 1 and isinstance(models[0], Mapping):
        items = list(models[0].items())
    else:
        items = [(None, m) for m in models]
    out: list[tuple[str, Any]] = []
    for name, model in items:
        if not _is_model(model):
            raise TypeError(f"modelsummary(): expected models from lm(), glm() or another stats3 fit, got {type(model).__name__}")
        if isinstance(model, Models):
            for fit in model:
                label = ", ".join(str(v) for v in fit.key.values())
                out.append((f"{name}: {label}" if name else label, fit))
        else:
            out.append((name, model))
    return [(name or f"({i})", fit) for i, (name, fit) in enumerate(out, start=1)]


def _display_terms(table: pl.DataFrame) -> pl.DataFrame:
    """Mixed-model variance components labelled as R's modelsummary does:
    ``SD (Intercept Subject)``, ``SD (Observations)``."""
    if "effect" not in table.columns:
        return table
    label = (pl.when(pl.col("effect") != "ran_pars").then(pl.col("term"))
             .when(pl.col("group") == "Residual").then(pl.lit("SD (Observations)"))
             .otherwise(pl.concat_str([pl.lit("SD ("), pl.col("term").str.replace("^sd__", ""),
                                       pl.lit(" "), pl.col("group"), pl.lit(")")])))
    return table.with_columns(label.alias("term"))


def _number(value: Any, digits: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def _stars(p: float | None) -> str:
    if p is None or math.isnan(p):
        return ""
    for cut, mark in STARS:
        if p < cut:
            return mark
    return ""


def modelsummary(
    *models: Any,
    statistic: str | None = "std_error",
    stars: bool = False,
    digits: int = 3,
    vcov: Any = None,
    conf_level: float = 0.95,
    exponentiate: bool = False,
    gof: tuple[str, ...] | list[str] = DEFAULT_GOF,
    output: str = "frame",
    as_: str = "polars",
) -> Any:
    """Models side by side, as R's ``modelsummary``.

    ``modelsummary(fit1, fit2)`` or ``modelsummary({"Base": fit1, "Full":
    fit2})``; a grouped fit gives one column per group. Each term shows its
    estimate and, below it, ``statistic``: ``"std_error"`` in parentheses
    (the default), ``"conf_int"`` as ``[low, high]``, ``"statistic"``,
    ``"p_value"``, or ``None`` for estimates only. ``stars=True`` marks
    p-values below 0.1, 0.05, 0.01 and 0.001 with ``+``, ``*``, ``**`` and
    ``***``. ``vcov`` (one for all models, or a list) uses robust standard
    errors, as in ``tidy``. ``gof`` picks the fit statistics, by ``glance``'s
    column names (those a model lacks stay empty).

    ``output="frame"`` gives a table with ``part`` (``"estimates"`` or
    ``"gof"``), ``term`` and ``statistic`` columns, then one text column per
    model, in ``as_``'s frame type; ``"markdown"`` and ``"html"`` give text.
    """
    from stats3.models import GeneralizedLinearModel, _Fit

    if statistic not in STATISTICS:
        raise ValueError(f"modelsummary(): statistic must be one of {list(STATISTICS)}, got {statistic!r}")
    if output not in ("frame", "markdown", "html"):
        raise ValueError(f"modelsummary(): output must be 'frame', 'markdown' or 'html', got {output!r}")
    if as_ not in OUTPUTS:
        raise ValueError(f"as_ must be one of {OUTPUTS}, got {as_!r}")
    named = _named(models)
    if not named:
        raise ValueError("modelsummary(): give at least one model")
    vcovs = list(vcov) if isinstance(vcov, (list, tuple)) else [vcov] * len(named)
    if len(vcovs) != len(named):
        raise ValueError(f"modelsummary(): {len(vcovs)} vcov for {len(named)} models")
    unknown = [g for g in gof if g not in GOF_LABELS]
    if unknown:
        raise ValueError(f"modelsummary(): unknown gof {unknown}; choose from {list(GOF_LABELS)}")

    tables, glances = [], []
    for (_, fit), v in zip(named, vcovs):
        assert isinstance(fit, _Fit)
        opts: dict[str, Any] = {"conf_int": statistic == "conf_int", "conf_level": conf_level, "vcov": v}
        if fit.kind == "multinomial":
            opts["exponentiate"] = exponentiate
        if isinstance(fit, GeneralizedLinearModel):
            opts["exponentiate"] = exponentiate
            if statistic == "conf_int" and v is None:
                opts["conf_method"] = "profile"
        table = _display_terms(fit.tidy(**opts))
        if "y_level" in table.columns:  # multinom: one row per response level and term
            table = table.with_columns(term=pl.format("{}:{}", "y_level", "term"))
        tables.append(table)
        glances.append(fit.glance(vcov=v) if v is not None and not isinstance(fit, GeneralizedLinearModel)
                       else fit.glance())

    terms: dict[str, None] = {}
    for table in tables:
        terms.update(dict.fromkeys(table["term"].to_list()))
    rows: list[dict[str, Any]] = []
    for term in terms:
        estimate = {"part": "estimates", "term": term, "statistic": "estimate"}
        below = {"part": "estimates", "term": term, "statistic": statistic or ""}
        for (name, _), table in zip(named, tables):
            found = table.filter(pl.col("term") == term)
            if found.height == 0:  # a term this model doesn't have
                estimate[name] = below[name] = ""
                continue
            row = found.row(0, named=True)
            text = _number(row["estimate"], digits)
            if stars and text:
                text += _stars(row["p_value"])
            estimate[name] = text
            if statistic == "std_error":
                below[name] = f"({_number(row['std_error'], digits)})" if row["std_error"] is not None else ""
            elif statistic == "conf_int":
                low, high = row["conf_low"], row["conf_high"]
                below[name] = (f"[{_number(low, digits)}, {_number(high, digits)}]"
                               if low is not None and high is not None else "")
            elif statistic is not None:
                below[name] = _number(row[statistic], digits)
        rows.append(estimate)
        if statistic is not None:
            rows.append(below)
    for key in gof:
        row = {"part": "gof", "term": GOF_LABELS[key], "statistic": ""}
        for (name, _), g in zip(named, glances):
            value = g[key][0] if key in g.columns else None
            row[name] = "" if value is None else (str(value) if key == "nobs" else _number(float(value), digits))
        if any(row[name] for name, _ in named):
            rows.append(row)
    if any(v is not None for v in vcovs):
        from stats3.robust import as_vcov

        row = {"part": "gof", "term": "Std.Errors", "statistic": ""}
        for (name, _), v in zip(named, vcovs):
            row[name] = "classical" if v is None else as_vcov(v).label
        rows.append(row)

    names = [name for name, _ in named]
    frame = pl.DataFrame(rows, schema={c: pl.String for c in ["part", "term", "statistic", *names]})
    if output == "frame":
        return Data(pl.LazyFrame(), [], {}, as_).write(frame)
    shown = [
        ["" if r["part"] == "estimates" and r["statistic"] != "estimate" else r["term"], *(r[n] for n in names)]
        for r in frame.iter_rows(named=True)
    ]
    notes = ["+ p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001"] if stars else []
    first_gof = next((i for i, r in enumerate(frame.iter_rows(named=True)) if r["part"] == "gof"), None)
    if output == "markdown":
        return _markdown(["", *names], shown, first_gof, notes)
    return _html(["", *names], shown, first_gof, notes)


def _markdown(header: list[str], rows: list[list[str]], rule: int | None, notes: list[str]) -> str:
    widths = [max(len(header[j]), *(len(r[j]) for r in rows)) for j in range(len(header))]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[0]) if j == 0 else c.rjust(widths[j])
                                 for j, c in enumerate(cells)) + " |"

    out = [line(header), "|" + "|".join(":" + "-" * (w + 1) if j == 0 else "-" * (w + 1) + ":"
                                        for j, w in enumerate(widths)) + "|"]
    for i, row in enumerate(rows):
        if i == rule:
            out.append(line(["-" * widths[0], *("-" * w for w in widths[1:])]))
        out.append(line(row))
    return "\n".join(out + [""] + notes) if notes else "\n".join(out)


def _html(header: list[str], rows: list[list[str]], rule: int | None, notes: list[str]) -> str:
    cell = html.escape
    head = "".join(f"<th>{cell(h)}</th>" for h in header)
    body = []
    for i, row in enumerate(rows):
        style = ' style="border-top: 1px solid"' if i == rule else ""
        tds = "".join(f'<td style="text-align: {"left" if j == 0 else "right"}">{cell(c)}</td>'
                      for j, c in enumerate(row))
        body.append(f"<tr{style}>{tds}</tr>")
    foot = "".join(f'<tr><td colspan="{len(header)}">{cell(n)}</td></tr>' for n in notes)
    foot = f"<tfoot>{foot}</tfoot>" if foot else ""
    return f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>\n" + "\n".join(body) + f"\n</tbody>{foot}\n</table>"
