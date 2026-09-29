"""Entity linking: find graph nodes mentioned in a question.

Longest-alias-first matching over node names and ``properties.aliases``;
each matched span is consumed so ``Apple Watch`` links to the product, not
to Apple Inc. plus a stray token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.graph.base import GraphStore, Node

_NORMALIZE_RE = re.compile(r"[^a-z0-9&+.' ]+")
_STOP_ALIASES = {"search", "core", "mac", "windows", "prime", "visa", "ford"}
_MIN_ALIAS_LEN = 3


def normalize(text: str) -> str:
    """Lower-case, strip punctuation except ``&+.'`` and collapse spaces."""
    return re.sub(r"\s+", " ", _NORMALIZE_RE.sub(" ", text.lower())).strip()


@dataclass(frozen=True)
class Mention:
    """A linked entity with its character span in the normalised question."""

    node_id: str
    alias: str
    start: int
    end: int


class EntityLinker:
    """Alias index over a :class:`GraphStore`."""

    def __init__(self, store: GraphStore) -> None:
        self._aliases: list[tuple[str, str]] = []
        self.rebuild(store)

    def rebuild(self, store: GraphStore) -> None:
        """Recompute the alias table from the current nodes."""
        pairs: dict[str, str] = {}
        for node in store.nodes():
            for alias in self._node_aliases(node):
                if len(alias) < 2:
                    continue
                # Prefer Company over other types when aliases collide (e.g. "visa").
                if alias in pairs and node["type"] != "Company":
                    continue
                pairs[alias] = node["id"]
        self._aliases = sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0]))

    @staticmethod
    def _node_aliases(node: Node) -> set[str]:
        aliases = {normalize(node["name"])}
        raw = node["properties"].get("aliases")
        if isinstance(raw, list):
            aliases.update(normalize(str(a)) for a in raw)
        ticker = node["properties"].get("ticker")
        if (
            node["type"] == "Company"
            and isinstance(ticker, str)
            and ticker.isalpha()
            and len(ticker) >= 2
        ):
            aliases.add(ticker.lower())
        name = normalize(node["name"])
        # "Apple Inc." -> "apple inc" ; also expose the name without legal suffixes
        stripped = re.sub(
            r"\b(inc|inc\.|incorporated|corporation|corp|corp\.|co|co\.|company|ltd|ltd\.|llc|plc|n\.v\.|holdings?|group|the)\b",
            " ",
            name,
        )
        stripped = re.sub(r"[,.]", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if len(stripped) >= _MIN_ALIAS_LEN and stripped not in _STOP_ALIASES:
            aliases.add(stripped)
        plurals = {
            a + "s"
            for a in aliases
            if a and not a.endswith("s") and " " in a or (a and len(a) > 4 and not a.endswith("s"))
        }
        return {a for a in aliases | plurals if a}

    def link(self, question: str) -> list[Mention]:
        """Return mentions in question order (non-overlapping, longest first)."""
        text = f" {normalize(question)} "
        taken = [False] * len(text)
        found: list[Mention] = []
        for alias, node_id in self._aliases:
            needle = f" {alias} "
            start = 0
            while True:
                idx = text.find(needle, start)
                if idx < 0:
                    break
                span = range(idx + 1, idx + 1 + len(alias))
                if not any(taken[i] for i in span):
                    for i in span:
                        taken[i] = True
                    found.append(
                        Mention(
                            node_id=node_id, alias=alias, start=idx + 1, end=idx + 1 + len(alias)
                        )
                    )
                start = idx + 1
        # possessive forms: "apple's"
        for alias, node_id in self._aliases:
            needle = f" {alias}'s "
            idx = text.find(needle)
            if idx >= 0 and not any(taken[idx + 1 : idx + 1 + len(alias)]):
                for i in range(idx + 1, idx + 1 + len(alias)):
                    taken[i] = True
                found.append(
                    Mention(node_id=node_id, alias=alias, start=idx + 1, end=idx + 1 + len(alias))
                )
        found.sort(key=lambda m: m.start)
        return found
