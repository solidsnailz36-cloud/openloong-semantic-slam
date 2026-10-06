#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One type-inference rule for every `key=value` parameter override.

Why this file exists
--------------------
The project has three places that turn command-line text into policy
parameters, and they have to agree:

    tools/online_pipeline.py   --explore-set KEY=VALUE
    tools/replay_explore.py    --set        KEY=VALUE
    tools/replay_explore.py    --arms       NAME:KEY=VALUE,KEY=VALUE

Two of them used `ast.literal_eval` and the third used an `int -> float -> str`
chain.  The chain **cannot produce a bool**, so

    --arms "off:rfirst=False"

arrived at the constructor as the *string* `"False"`, and `bool("False")` is
`True` -- the arm that was meant to be the control ran with the feature on.  Nothing
crashed, no warning was printed, and the comparison it produced was simply an
answer to a different question.  That is the same failure class this project has
already paid for twice ("a parameter that silently failed to apply"), which is
why the fix is not a patch to one parser but one parser for all of them.

`literal_eval` is the rule because it is the only one that round-trips the way a
Python user writes: `False` is a bool, `0.6` is a float, `1` is an int, `add` is a
string.  A value it cannot parse is the caller's mistake and is left as text, so a
mistyped number still raises inside the comparison it feeds rather than at the
command line -- with `--explore-set`-style flags that is the intended behaviour,
because "the flag silently did nothing" is what the flag exists to catch.
"""
from __future__ import annotations

import ast


def parse_kv_overrides(items) -> dict:
    """`["a=1", "b=False"]` -> `{"a": 1, "b": False}`.

    Raises on an item with no `=`: a bare word was meant to be something and
    silently dropping it is the failure mode this function exists to prevent.
    """
    out: dict = {}
    for it in items or ():
        if "=" not in it:
            raise SystemExit("expected key=value, got %r" % (it,))
        k, v = it.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def as_bool(v) -> bool:
    """A defensive reader for a flag that may have arrived as text.

    `bool("False")` is `True`, and `bool("0")` is `True`, so a flag that is read
    as `bool(x)` somewhere downstream is one careless caller away from being
    stuck on.  Callers that take a flag from the command line should use this;
    `parse_kv_overrides` already returns real bools, so this only matters for a
    path that skipped it.
    """
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off", "none")
    return bool(v)
