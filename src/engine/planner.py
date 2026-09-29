"""Question → traversal plan.

The planner is deterministic and inspectable (the executed plan is returned
to the client). It maps relation cue phrases to edge types, attaches cue
groups to the entity mentions around them, and resolves edge direction from
the edge signatures at execution time; only ``Company→Company`` edges need a
phrase-level direction hint.

Grammar handled (see ``eval/gold.jsonl`` for 100 examples):

* ``<cues> <seed>``            → cues nest outward, applied in reverse order
* ``<seed>'s <cues>``          → cues applied forward
* ``<cues1> <seedA> <cues2> <seedB>`` → constraint sets intersected
* ``share/same/common <relation>`` → out-and-back through that relation
* ``how many ...``             → same plan, count the final frontier
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.engine.linking import Mention, normalize
from src.extraction.schemas import EDGE_SIGNATURES

Direction = Literal["in", "out", "both", "auto"]

#: (regex over the normalised question, edge types, direction hint)
RELATION_CUES: list[tuple[str, tuple[str, ...], Direction]] = [
    (
        r"\b(ceos?|chief executives?|runs?|lead(?:s|er)?|led by|head(?:s|ed)? by)\b",
        ("CEO_OF",),
        "auto",
    ),
    (r"\b(directors?|boards? members?|boards?|sits? on)\b", ("DIRECTOR_OF",), "auto"),
    (r"\b(parent(?: compan(?:y|ies))?|owner|owned by|belongs? to)\b", ("SUBSIDIARY_OF",), "auto"),
    (r"\b(subsidiar(?:y|ies)|units?|affiliates?|owns?)\b", ("SUBSIDIARY_OF",), "auto"),
    (
        r"\b(supplied by|depend(?:s|ent|ing)? on|rel(?:y|ies|ying) on|customers? of|buys? from|sources? from)\b",
        ("SUPPLIED_BY",),
        "in",
    ),
    (r"\b(suppliers?(?: of)?|supply chain|vendors?|supplied to)\b", ("SUPPLIED_BY",), "out"),
    (
        r"\b(competitors?|compet(?:e|es|ing|ition) (?:with|against)|rivals?)\b",
        ("COMPETES_WITH",),
        "both",
    ),
    (
        r"\b(risks?|risk factors?|exposures?|exposed to|mention(?:s|ed|ing)?)\b",
        ("MENTIONS_RISK",),
        "auto",
    ),
    (
        r"\b(products?|offer(?:s|ed|ings?)?|sells?|makes?|drugs?|vehicles?)\b",
        ("OFFERS_PRODUCT",),
        "auto",
    ),
    (r"\b(markets?|operates? in|industr(?:y|ies)|segments?)\b", ("OPERATES_IN",), "auto"),
]
_SHARE_RE = re.compile(r"\b(share|shares|same|common|also)\b")
_COUNT_RE = re.compile(r"\b(how many|count|number of|total)\b")
_MULTI_RE = re.compile(r"\b(more than one|at least two|two or more|multiple|several)\b")
_SP500_RE = re.compile(r"\bs&p 500\b|\bs&p\b|\bsp500\b")
_SECTOR_RE = re.compile(r"\bin the ([a-z&+ ]+?) sector\b")
_PROPERTY_RE = re.compile(
    r"\b(ticker|ticker symbol|sector|cik|headquarter(?:s|ed)?|hq|state|jurisdiction|incorporated|since|category|role)\b"
)
_ANSWER_TYPE_RE: list[tuple[str, str]] = [
    (r"\b(who|which (?:people|persons?|executives?|ceos?|directors?|leaders?)|whose)\b", "Person"),
    (
        r"\b(which|what) (?:s&p 500 )?(?:compan(?:y|ies)|firms?|filers?|customers?|suppliers?|competitors?|rivals?|banks?|automakers?)\b",
        "Company",
    ),
    (r"\b(which|what) (?:risks?|risk factors?|exposures?)\b", "Risk"),
    (r"\b(which|what) (?:products?|drugs?|offerings?|vehicles?|brands?)\b", "Product"),
    (r"\b(which|what) (?:subsidiar(?:y|ies)|units?|entities)\b", "Subsidiary"),
    (r"\b(which|what) (?:markets?|industr(?:y|ies)|segments?)\b", "Market"),
    (
        r"\bhow many (?:s&p 500 )?(?:compan(?:y|ies)|firms?|filers?|customers?|suppliers?|competitors?)\b",
        "Company",
    ),
    (r"\bhow many (?:people|persons?|executives?|ceos?|directors?)\b", "Person"),
    (r"\bhow many (?:risks?|risk factors?)\b", "Risk"),
    (r"\bhow many (?:products?|drugs?|offerings?|vehicles?)\b", "Product"),
    (r"\bhow many (?:subsidiar(?:y|ies)|units?)\b", "Subsidiary"),
    (r"\bhow many (?:markets?|industr(?:y|ies))\b", "Market"),
]
_ROLE_RE: list[tuple[str, str]] = [
    (r"\b(ceos?|chief executives?)\b", "CEO"),
    (r"\b(directors?|board)\b", "Director"),
]
_PROPERTY_ALIASES = {
    "ticker symbol": "ticker",
    "headquarters": "hq_state",
    "headquartered": "hq_state",
    "headquarter": "hq_state",
    "hq": "hq_state",
    "state": "hq_state",
    "incorporated": "jurisdiction",
}


@dataclass
class Step:
    """One traversal hop."""

    edge_types: tuple[str, ...]
    direction: Direction
    share: bool = False  # out-and-back through the same relation

    def as_dict(self) -> dict[str, object]:
        """Serializable view."""
        return {
            "edge_types": list(self.edge_types),
            "direction": self.direction,
            "share": self.share,
        }


@dataclass
class Constraint:
    """A seed with the steps that expand it."""

    seed_id: str
    steps: list[Step]


@dataclass
class QueryPlan:
    """Everything the executor needs."""

    question: str
    constraints: list[Constraint]
    aggregate: bool
    answer_type: str | None
    role_filter: str | None
    in_sp500_only: bool
    sector: str | None
    property_lookup: str | None
    kind: str = "lookup"
    notes: list[str] = field(default_factory=list)
    seed_types: dict[str, str] = field(default_factory=dict)
    min_degree: int | None = None
    seedless_steps: list[Step] = field(default_factory=list)

    @property
    def n_hops(self) -> int:
        """Maximum hop count across constraints (share steps count as 2)."""
        best = 0
        for c in self.constraints:
            hops = sum(2 if s.share else 1 for s in c.steps)
            best = max(best, hops)
        return best

    @property
    def seed_ids(self) -> list[str]:
        """Unique seeds in order."""
        seen: list[str] = []
        for c in self.constraints:
            if c.seed_id not in seen:
                seen.append(c.seed_id)
        return seen

    def as_dict(self) -> dict[str, object]:
        """Serializable view."""
        return {
            "kind": self.kind,
            "aggregate": self.aggregate,
            "answer_type": self.answer_type,
            "role_filter": self.role_filter,
            "in_sp500_only": self.in_sp500_only,
            "sector": self.sector,
            "property_lookup": self.property_lookup,
            "constraints": [
                {"seed": c.seed_id, "steps": [s.as_dict() for s in c.steps]}
                for c in self.constraints
            ],
            "notes": self.notes,
        }


@dataclass
class _Cue:
    start: int
    end: int
    step: Step


def _find_cues(text: str) -> list[_Cue]:
    cues: list[_Cue] = []
    taken: list[bool] = [False] * (len(text) + 1)
    for pattern, edge_types, direction in RELATION_CUES:
        for m in re.finditer(pattern, text):
            if any(taken[m.start() : m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            cues.append(_Cue(m.start(), m.end(), Step(edge_types, direction)))
    cues.sort(key=lambda c: c.start)
    # attach share/same/common to the cue that follows the marker
    for m in _SHARE_RE.finditer(text):
        following = [c for c in cues if c.start >= m.end()]
        if following:
            following[0].step.share = True
    return cues


def _answer_type(text: str) -> str | None:
    for pattern, node_type in _ANSWER_TYPE_RE:
        if re.search(pattern, text):
            return node_type
    return None


def _role_filter(text: str, answer_type: str | None) -> str | None:
    if answer_type != "Person":
        return None
    for pattern, role in _ROLE_RE:
        if re.search(pattern, text):
            return role
    return None


def build_plan(
    question: str, mentions: list[Mention], seed_types: dict[str, str] | None = None
) -> QueryPlan:
    """Build a :class:`QueryPlan` from the question and its linked mentions.

    Args:
        question: Raw question.
        mentions: Output of :meth:`~src.engine.linking.EntityLinker.link`.
        seed_types: ``node_id -> type`` for the mentioned nodes (lets the planner
            derive the answer type from edge signatures).
    """
    text = normalize(question)
    cues = _find_cues(text)
    notes: list[str] = []
    aggregate = bool(_COUNT_RE.search(text))
    answer_type = _answer_type(text)
    role_filter = _role_filter(text, answer_type)
    in_sp500 = bool(_SP500_RE.search(text))
    sector_m = _SECTOR_RE.search(text)
    sector = sector_m.group(1).strip() if sector_m else None

    prop_m = _PROPERTY_RE.search(text)
    property_lookup = None
    if prop_m and not cues and mentions:
        raw = prop_m.group(1)
        property_lookup = _PROPERTY_ALIASES.get(raw, raw)

    constraints: list[Constraint] = []
    # Entity mentions win over relation cues ("drug" inside "Drug pricing pressure").
    cues = [c for c in cues if not any(m.start < c.end and c.start < m.end for m in mentions)]
    if len(mentions) >= 2:
        for c in cues:
            c.step.share = False  # "shared by Ford and GM" is an intersection, not out-and-back
    if not mentions:
        notes.append("no entity mention linked; seedless plan")
    min_degree = 2 if _MULTI_RE.search(text) else None
    seedless_steps = _collapse([c.step for c in cues]) if not mentions else []
    for i, mention in enumerate(mentions):
        prev_end = mentions[i - 1].end if i > 0 else -1
        next_start = mentions[i + 1].start if i + 1 < len(mentions) else len(text) + 1
        before = [c.step for c in cues if prev_end < c.start < mention.start]
        # cues after the LAST mention belong to it; cues between mentions belong to the next one
        after = (
            [c.step for c in cues if mention.end <= c.start < next_start]
            if i == len(mentions) - 1
            else []
        )
        steps = _collapse(list(reversed(before)) + after)
        if not steps and constraints and i > 0:
            # "both Pfizer and Merck": the second seed inherits the first seed's steps
            steps = [Step(s.edge_types, s.direction, s.share) for s in constraints[-1].steps]
            notes.append(f"{mention.node_id} inherits steps from {constraints[-1].seed_id}")
        constraints.append(Constraint(seed_id=mention.node_id, steps=steps))

    plan = QueryPlan(
        question=question,
        constraints=constraints,
        aggregate=aggregate,
        answer_type=answer_type,
        role_filter=role_filter,
        in_sp500_only=in_sp500,
        sector=sector,
        property_lookup=property_lookup,
        notes=notes,
        seed_types=dict(seed_types or {}),
        min_degree=min_degree,
        seedless_steps=seedless_steps,
    )
    derived = derived_answer_type(plan)
    if derived is not None:
        plan.answer_type = derived
    plan.role_filter = _role_filter(text, plan.answer_type)
    plan.kind = classify_plan(plan)
    return plan


def _collapse(steps: list[Step]) -> list[Step]:
    """Merge consecutive steps on the same edge types ("products ... offer")."""
    out: list[Step] = []
    for s in steps:
        if out and out[-1].edge_types == s.edge_types and not (out[-1].share or s.share):
            if out[-1].direction == "auto":
                out[-1].direction = s.direction
            continue
        out.append(s)
    return out


def derived_answer_type(plan: QueryPlan) -> str | None:
    """Node type at the far end of the last step, from the edge signatures.

    ``None`` when there are no steps (property lookup / bare mention) or when
    the last step's edge types disagree.
    """
    types: set[str] = set()
    for c in plan.constraints:
        if not c.steps:
            continue
        last = c.steps[-1]
        for et in last.edge_types:
            src, dst = EDGE_SIGNATURES[et]
            if src == dst or last.share:
                types.add(src)
            elif last.direction == "in":
                types.add(src)
            elif last.direction == "out":
                types.add(dst)
            else:
                # auto: whichever end is NOT the previous frontier type; resolved
                # via the previous step when possible, else the question noun.
                prev_type = _frontier_type_before(plan, c, len(c.steps) - 1)
                if prev_type == src:
                    types.add(dst)
                elif prev_type == dst:
                    types.add(src)
    return next(iter(types)) if len(types) == 1 else None


def _frontier_type_before(plan: QueryPlan, c: Constraint, idx: int) -> str | None:
    """Type of the frontier entering step ``idx`` (seed type when idx == 0)."""
    node_type = plan.seed_types.get(c.seed_id)
    for s in c.steps[:idx]:
        et = s.edge_types[0]
        src, dst = EDGE_SIGNATURES[et]
        if s.share or src == dst:
            continue
        node_type = dst if node_type == src else src
    return node_type


def classify_plan(plan: QueryPlan) -> str:
    """Kind label derived from the plan structure."""
    if plan.aggregate:
        return "aggregation"
    if not plan.constraints:
        return "lookup" if plan.answer_type else "hybrid"
    if plan.n_hops <= 1:
        return "lookup"
    return "multi_hop"


def resolve_direction(step: Step, frontier_types: set[str]) -> Literal["in", "out", "both"]:
    """Resolve ``auto`` using edge signatures and the types on the frontier."""
    if step.direction != "auto":
        return step.direction
    outs = 0
    ins = 0
    for et in step.edge_types:
        src, dst = EDGE_SIGNATURES[et]
        if src in frontier_types:
            outs += 1
        if dst in frontier_types:
            ins += 1
    if outs and not ins:
        return "out"
    if ins and not outs:
        return "in"
    return "both"
