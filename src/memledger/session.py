"""Session API for observation, recall, context building and checkpointing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memledger.checkpoint import CheckpointReport, run_checkpoint
from memledger.events import Cause, Event, make_event
from memledger.ids import new_id
from memledger.retrieval import retrieve
from memledger.tuples import MemoryTuple, make_tuple

if TYPE_CHECKING:
    from memledger.api import Ledger


@dataclass(slots=True)
class Context:
    system: str
    messages: list[dict[str, str]]


def _observed_turn(event: Event) -> int:
    try:
        return int(event.payload.get("turn", 0))
    except (TypeError, ValueError):
        return 0


def _record_source_events(ledger: Ledger, record: MemoryTuple) -> list[Event]:
    observed: list[Event] = []
    for source_id in record.sources:
        event = ledger.store.get_event(source_id)
        if event is None or event.type != "observed":
            continue
        observed.append(event)
    observed.sort(key=lambda event: (event.ts, _observed_turn(event), event.id))
    return observed


def _record_chronology_key(ledger: Ledger, record: MemoryTuple) -> tuple[str, str, int, str]:
    observed = _record_source_events(ledger, record)
    source_ts = observed[0].ts if observed else record.created_ts
    primary_ts = str(record.qualifiers.get("when", "")).strip() or source_ts
    source_turn = _observed_turn(observed[0]) if observed else 0
    return (primary_ts, source_ts, source_turn, record.id)


def _render_context_record(ledger: Ledger, record: MemoryTuple) -> str:
    observed = _record_source_events(ledger, record)
    details: list[str] = []
    if observed:
        details.append(f"observed {observed[0].ts}")
        turn = _observed_turn(observed[0])
        if turn:
            details.append(f"turn {turn}")
    elif record.created_ts:
        details.append(f"recorded {record.created_ts}")
    if not details:
        return f"- {record.text_form}"
    return f"- [{' | '.join(details)}] {record.text_form}"


class Session:
    """High-level session facade backed by the append-only ledger."""

    def __init__(
        self,
        ledger: Ledger,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.ledger = ledger
        self.id = session_id or new_id("se")
        self.user_id = user_id

    def observe(self, *, user: str, assistant: str) -> None:
        for role, text in (("user", user), ("assistant", assistant)):
            turn = self.ledger.store.next_turn(self.id)
            event = make_event(
                type="observed",
                actor="dev",
                cause=Cause(kind="signal", ref="observe", detail="session observe"),
                policy_hash=self.ledger.policy.hash,
                payload={"role": role, "text": text, "turn": turn},
                session=self.id,
                user=self.user_id,
            )
            self.ledger.append_event(event)

    def feedback(self, value: int, *, on: str = "last_turn") -> None:
        target = on
        if on == "last_turn":
            observed = self.ledger.store.iter_events(session=self.id, type="observed")
            if not observed:
                raise ValueError("no observed turns in session")
            target = observed[-1].id
        event = make_event(
            type="feedback",
            actor="dev",
            cause=Cause(kind="signal", ref="feedback", detail="explicit feedback"),
            policy_hash=self.ledger.policy.hash,
            payload={"value": value, "on": target},
            session=self.id,
            user=self.user_id,
        )
        self.ledger.append_event(event)

    def outcome(self, status: str, *, task: str, on: list[str] | None = None) -> None:
        if on is None:
            observed = self.ledger.store.iter_events(session=self.id, type="observed")
            on = [observed[-1].id] if observed else []
        event = make_event(
            type="outcome",
            actor="dev",
            cause=Cause(kind="signal", ref="outcome", detail=f"declared {status}"),
            policy_hash=self.ledger.policy.hash,
            payload={"status": status, "task": task, "on": on},
            session=self.id,
            user=self.user_id,
        )
        self.ledger.append_event(event)

    def remember(self, tuple_value: tuple[str, str, str | int | float | bool]) -> str:
        subject, relation, value = tuple_value
        record = make_tuple(
            subject=subject,
            relation=relation,
            value=value,
            qualifiers={},
            confidence=1.0,
            layer="episodic",
            status="active",
            ttl=self.ledger.policy.ttl_for_layer("episodic"),
            sessions_seen=[self.id],
            sources=[],
        )
        event = make_event(
            type="remember",
            actor="dev",
            cause=Cause(kind="manual", ref="remember", detail="developer remembered tuple"),
            policy_hash=self.ledger.policy.hash,
            payload={"tuple": record.to_dict()},
            session=self.id,
            user=self.user_id,
        )
        self.ledger.append_event(event)
        return record.id

    def recall(self, query: str, k: int = 5) -> list[MemoryTuple]:
        return retrieve(self, query, k)

    def build_context(
        self,
        *,
        instinct: bool,
        episodic: Iterable[MemoryTuple],
        working: str,
    ) -> Context:
        if working != "tail":
            raise ValueError("only working='tail' is supported in MemLedger 0.1")
        messages: list[dict[str, str]] = []
        token_budget = self.ledger.policy.working_token_budget
        observed = self.ledger.store.iter_events(session=self.id, type="observed")
        used = 0
        for event in reversed(observed):
            text = str(event.payload["text"])
            tokens = len(text.split())
            if used + tokens > token_budget:
                break
            messages.append({"role": str(event.payload["role"]), "content": text})
            used += tokens
        messages.reverse()

        system_lines = []
        if instinct:
            instinct_records = sorted(
                self.ledger.projection.active_instinct(),
                key=lambda record: (-record.impact, record.id),
            )[: int(self.ledger.policy.get("instinct", "max_items", default=30))]
            if instinct_records:
                system_lines.append("Instinct memory:")
                system_lines.extend(f"- {record.text_form}" for record in instinct_records)
        episodic_records = sorted(
            list(episodic),
            key=lambda record: _record_chronology_key(self.ledger, record),
        )
        if episodic_records:
            system_lines.append("Relevant episodic memory:")
            system_lines.extend(_render_context_record(self.ledger, record) for record in episodic_records)
        return Context(system="\n".join(system_lines), messages=messages)

    def checkpoint(self) -> CheckpointReport:
        return run_checkpoint(self)

    def working_summary(self) -> str:
        observed = self.ledger.store.iter_events(session=self.id, type="observed")
        tail = observed[-4:]
        return " | ".join(f"{event.payload['role']}: {event.payload['text']}" for event in tail)
