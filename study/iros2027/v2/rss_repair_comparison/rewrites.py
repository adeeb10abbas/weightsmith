"""Frozen, conservative text-only routing for a two-entry clarification memory.

This intentionally gives the rewrite comparator a semantic parser, not an exact
string lookup. It receives no benchmark cell or task ID. Uncertain requests and
different intentions pass through unchanged. Rules and vocabulary are explicit
baseline engineering; no learned language-generalization claim is made for it.
"""
from __future__ import annotations

import re

TEMPORAL_SOURCE = "Put the cube then the banana in the bowl"
TEMPORAL_TARGET = "Put the cube in the bowl then put the banana in the bowl"
SPATIAL_SOURCE = "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
SPATIAL_TARGET = "Move the mustard bottle to the right of the raisin box on the table. Leave the raisin box where it is."


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def intent(text):
    s = normalize(text)
    # Negation/alternatives and unfamiliar qualifications are safer to abstain on.
    if re.search(r"\b(not|never|instead|unless|either|without)\b", s):
        return None
    if all(w in s for w in ("cube", "banana", "bowl")):
        if re.search(r"\bonly\b|\bleave\b.*\bbanana\b", s):
            return None
        for marker, reverse in (("before", False), ("after", True), ("then", False), ("next", False)):
            if re.search(rf"\b{marker}\b", s):
                a, b = re.split(rf"\b{marker}\b", s, maxsplit=1)
                order = ("cube", "banana") if "cube" in a and "banana" in b else (
                    ("banana", "cube") if "banana" in a and "cube" in b else None)
                if order and not ("banana" in a and "cube" in a):
                    if reverse:
                        order = order[::-1]
                    return "cube_then_banana" if order == ("cube", "banana") else "banana_then_cube"
        return None
    if "mustard" in s and "raisin" in s:
        if re.search(r"\bon top\b|\bstack\b|\bon the raisin\b", s):
            return "mustard_on_raisin"
        # Canonicalize relational statements including reference-first clauses.
        clauses = re.findall(r"(mustard(?: bottle)?|raisin(?: box)?)(?:(?!mustard|raisin).){0,55}?\b(left|right)\b(?:(?!mustard|raisin).){0,25}?(mustard(?: bottle)?|raisin(?: box)?)", s)
        meanings = set()
        for subject, relation, reference in clauses:
            if subject.startswith("mustard") and reference.startswith("raisin"):
                meanings.add("mustard_" + relation)
            elif subject.startswith("raisin") and reference.startswith("mustard"):
                meanings.add("mustard_" + ("right" if relation == "left" else "left"))
        # Never convert an instruction that explicitly moves the reference object.
        if re.search(r"\b(move|put|place|slide|set|position)\s+(?:only\s+)?(?:the\s+)?raisin", s):
            return None
        return next(iter(meanings)) if len(meanings) == 1 else None
    return None


def route(text, mode="semantic"):
    if mode == "exact":
        memory = {normalize(TEMPORAL_SOURCE): TEMPORAL_TARGET,
                  normalize(SPATIAL_SOURCE): SPATIAL_TARGET}
        effective = memory.get(normalize(text), text)
        key = "exact" if effective != text else None
    elif mode == "semantic":
        key = intent(text)
        effective = {"cube_then_banana": TEMPORAL_TARGET,
                     "mustard_right": SPATIAL_TARGET}.get(key, text)
    else:
        raise ValueError(mode)
    return {"input_prompt": text, "effective_prompt": effective,
            "route": key, "rewritten": effective != text, "mode": mode}
