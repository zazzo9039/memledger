"""Command-line interface for MemLedger."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from memledger.api import Ledger
from memledger.prompts import find_project_root

_TREE_CHILD = "\u2514\u2500"
_TREE_INDENT = "   "


def _default_db_path() -> str:
    return "memory.db"


def _copy_defaults(target_dir: Path) -> None:
    root = find_project_root()
    policy_target = target_dir / "memory.policy.yaml"
    prompts_target = target_dir / "prompts"
    if not policy_target.exists():
        shutil.copy2(root / "memory.policy.yaml", policy_target)
    prompts_target.mkdir(exist_ok=True)
    for prompt_file in (root / "prompts").iterdir():
        if prompt_file.is_file() and not (prompts_target / prompt_file.name).exists():
            shutil.copy2(prompt_file, prompts_target / prompt_file.name)


def _quote_text(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _event_date(value: str | None) -> str | None:
    if not value:
        return None
    return value.split("T", 1)[0]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _render_tree_line(
    indent: str,
    label: str,
    date: str | None = None,
    details: list[str] | None = None,
) -> str:
    line = f"{indent}{_TREE_CHILD} {label}"
    if date:
        line += f"  {date}"
    if details:
        rendered = [detail for detail in details if detail]
        if rendered:
            line += "  " + "  ".join(rendered)
    return line


def _format_why_header(record: dict[str, Any]) -> str:
    return (
        f"{record['id']}  ({record['layer']}, {record['status']})  "
        f"{_quote_text(str(record.get('text_form', '')))}"
    )


def _find_creator_tuple_payload(creator: dict[str, Any], record_id: str) -> dict[str, Any] | None:
    payload = creator.get("payload", {})
    if creator.get("type") == "remember":
        record = payload.get("tuple")
        return record if isinstance(record, dict) else None
    for record in payload.get("tuples", []):
        if isinstance(record, dict) and str(record.get("id")) == record_id:
            return record
    return None


def _format_why_lifecycle(ledger: Ledger, event: Any) -> tuple[str, str | None, list[str]] | None:
    date = _event_date(getattr(event, "ts", None))
    if event.type == "promotion_approved":
        proposal_id = str(event.payload.get("proposal", "")).strip()
        proposal = ledger.store.get_event(proposal_id) if proposal_id else None
        rationale = ""
        if proposal is not None:
            rationale = str(proposal.payload.get("rationale") or proposal.cause.detail or "").strip()
        else:
            rationale = str(event.cause.detail or "").strip()
        approved_by = str(event.payload.get("by", "")).strip()
        details = []
        if rationale and approved_by:
            details.append(f"cause: {rationale}, approved by {approved_by}")
        elif rationale:
            details.append(f"cause: {rationale}")
        elif approved_by:
            details.append(f"approved by {approved_by}")
        return ("promoted", date, details)
    if event.type == "quarantine_lifted":
        sessions = _safe_int(event.payload.get("confirmed_in_sessions"))
        details = [f"confirmed in {sessions} sessions"] if sessions else []
        return ("confirmed", date, details)
    if event.type == "merged":
        reason = str(event.cause.detail or "").strip()
        merged_from = [str(value) for value in event.payload.get("from", [])]
        details = []
        if reason:
            details.append(f"cause: {reason}")
        if merged_from:
            details.append(f"from: {', '.join(merged_from)}")
        return ("merged", date, details)
    if event.type == "superseded":
        reason = str(event.payload.get("reason") or event.cause.detail or "").strip()
        replacement = str(event.payload.get("new", "")).strip()
        details = []
        if reason:
            details.append(f"cause: {reason}")
        if replacement:
            details.append(f"by: {replacement}")
        return ("superseded", date, details)
    if event.type == "deleted":
        reason = str(event.payload.get("reason") or event.cause.detail or "").strip()
        details = [f"cause: {reason}"] if reason else []
        return ("deleted", date, details)
    if event.type == "expired":
        return ("expired", date, [])
    if event.type == "regenerated":
        return ("regenerated", date, [])
    return None


def _collect_why_lifecycle(ledger: Ledger, why_data: dict[str, Any]) -> list[tuple[str, str | None, list[str]]]:
    creator = why_data.get("creator") or {}
    creator_id = str(creator.get("id", "")).strip()
    history_ids = [
        str(event_id)
        for event_id in why_data.get("history", [])
        if str(event_id).strip() and str(event_id) != creator_id
    ]
    history_events = [ledger.store.get_event(event_id) for event_id in history_ids]
    resolved = [event for event in history_events if event is not None]
    has_promotion = any(event.type == "promotion_approved" for event in resolved)
    lifecycle: list[tuple[str, str | None, list[str]]] = []
    for event in resolved:
        if event.type in {"scored", "extracted"}:
            continue
        if has_promotion and event.type == "quarantine_lifted":
            continue
        rendered = _format_why_lifecycle(ledger, event)
        if rendered is not None:
            lifecycle.append(rendered)
    lifecycle.reverse()
    return lifecycle


def _format_why_creator(record_id: str, creator: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    event_type = str(creator.get("type", "origin")).strip() or "origin"
    date = _event_date(str(creator.get("ts", "")).strip())
    if event_type == "remember":
        details = []
        session_id = str(creator.get("session", "")).strip()
        if session_id:
            details.append(f"session: {session_id}")
        return ("remembered", date, details)
    if event_type == "seeded":
        return ("seeded", date, [])
    if event_type == "extracted":
        details = []
        llm = creator.get("llm", {})
        model = str(llm.get("model", "")).strip()
        prompt = str(llm.get("prompt", "")).strip()
        tuple_payload = _find_creator_tuple_payload(creator, record_id)
        confidence = tuple_payload.get("confidence") if tuple_payload else None
        if model:
            details.append(f"model: {model}")
        if prompt:
            details.append(f"prompt: {prompt}")
        if confidence is not None:
            details.append(f"confidence: {confidence}")
        return ("extracted", date, details)
    return (event_type, date, [])


def _sorted_why_sources(why_data: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [source for source in why_data.get("sources", []) if isinstance(source, dict)]
    return sorted(
        sources,
        key=lambda source: (
            str(source.get("ts", "")),
            str(source.get("session", "")),
            _safe_int(source.get("payload", {}).get("turn")),
            str(source.get("id", "")),
        ),
    )


def _format_why_source(indent: str, source: dict[str, Any]) -> str:
    event_type = str(source.get("type", "source")).strip() or "source"
    if event_type == "observed":
        payload = source.get("payload", {})
        details = []
        session_id = str(source.get("session", "")).strip()
        turn = payload.get("turn")
        if session_id and turn is not None:
            details.append(f"{session_id} turn {turn}")
        elif session_id:
            details.append(session_id)
        elif turn is not None:
            details.append(f"turn {turn}")
        text = str(payload.get("text", "")).strip()
        if text:
            details.append(_quote_text(text))
        return _render_tree_line(indent, "observed", details=details)
    return _render_tree_line(indent, event_type, _event_date(str(source.get("ts", "")).strip()))


def _format_why_text(ledger: Ledger, why_data: dict[str, Any]) -> str:
    record = why_data["record"]
    lines = [_format_why_header(record)]
    indent = " "
    for label, date, details in _collect_why_lifecycle(ledger, why_data):
        lines.append(_render_tree_line(indent, label, date, details))
        indent += _TREE_INDENT

    creator = why_data.get("creator")
    creator_id = ""
    if isinstance(creator, dict) and creator:
        creator_id = str(creator.get("id", "")).strip()
        label, date, details = _format_why_creator(str(record["id"]), creator)
        lines.append(_render_tree_line(indent, label, date, details))
        indent += _TREE_INDENT

    for source in _sorted_why_sources(why_data):
        if creator_id and str(source.get("id", "")).strip() == creator_id:
            continue
        lines.append(_format_why_source(indent, source))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memledger")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("path", nargs="?", default=_default_db_path())

    log_parser = subparsers.add_parser("log")
    log_parser.add_argument("--db", default=_default_db_path())
    log_parser.add_argument("--type")
    log_parser.add_argument("--session")
    log_parser.add_argument("--since")

    why_parser = subparsers.add_parser("why")
    why_parser.add_argument("id")
    why_parser.add_argument("--db", default=_default_db_path())
    why_parser.add_argument("--json", action="store_true")

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--db", default=_default_db_path())

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--db", default=_default_db_path())
    replay_parser.add_argument("--at")
    replay_parser.add_argument("--cached", action="store_true")

    rebuild_parser = subparsers.add_parser("rebuild")
    rebuild_parser.add_argument("--db", default=_default_db_path())

    regen_parser = subparsers.add_parser("regenerate")
    regen_parser.add_argument("--db", default=_default_db_path())
    regen_parser.add_argument("--model")
    regen_parser.add_argument("--prompt", default="extract@v1")

    delete_parser = subparsers.add_parser("delete")
    delete_parser.add_argument("id")
    delete_parser.add_argument("--db", default=_default_db_path())
    delete_parser.add_argument("--cascade", action="store_true")
    delete_parser.add_argument("--reason", default="manual")

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--db", default=_default_db_path())
    return parser


def _open_ledger(path: str) -> Ledger:
    return Ledger(path=path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "init":
        target_path = Path(args.path)
        _copy_defaults(target_path.parent if target_path.parent != Path("") else Path.cwd())
        ledger = Ledger(path=str(target_path))
        ledger.close()
        print(target_path)
        return 0

    ledger = _open_ledger(args.db)
    try:
        if command == "log":
            events = ledger.store.iter_events(session=args.session, type=args.type, since=args.since)
            for event in events:
                print(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0

        if command == "why":
            why_data = ledger.why(args.id)
            if args.json:
                print(json.dumps(why_data, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(_format_why_text(ledger, why_data))
            return 0

        if command == "review":
            proposals = [event for event in ledger.store.iter_events(type="promotion_proposed")]
            print(
                json.dumps(
                    [event.to_dict() for event in proposals],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if command == "replay":
            ledger.replay(at=args.at, cached=args.cached)
            print("ok")
            return 0

        if command == "rebuild":
            ok = ledger.rebuild()
            print("ok" if ok else "mismatch")
            return 0 if ok else 1

        if command == "regenerate":
            count = ledger.regenerate(model=args.model, prompt=args.prompt)
            print(count)
            return 0

        if command == "delete":
            ledger.delete(args.id, cascade=args.cascade, reason=args.reason)
            print(args.id)
            return 0

        if command == "stats":
            print(json.dumps(ledger.stats(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    finally:
        ledger.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
