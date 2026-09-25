from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import anyio
import httpx2
from mcp.shared.exceptions import MCPError

from .cases import load_case_set
from .config import Settings
from .contracts import Contracts
from .mcp_gateway import EvidenceGateway, MCPTimeoutError, connect_gateway
from .submission import package_submission, validate_artifacts
from .trace import TraceWriter
from .workflow import solve_case


def _root(value: str) -> Path:
    return Path(value).resolve()


async def _show_tools(root: Path) -> None:
    settings = Settings.load(root)
    contracts = Contracts(root / "contracts" / "schemas")
    async with connect_gateway(settings.mcp_endpoint, settings.team_api_key, contracts) as gateway:
        for tool in await gateway.list_tools():
            print(tool)


MAX_CONNECTION_ATTEMPTS = 6
MAX_RETRY_DELAY_SECONDS = 60
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    httpx2.TransportError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    MCPTimeoutError,
    MCPError,
    TimeoutError,
    ConnectionError,
)


def _is_transient(exc: BaseException) -> bool:
    """Network/session failures that a reconnect can fix; logic errors are never retried."""
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_transient(inner) for inner in exc.exceptions)
    return isinstance(exc, TRANSIENT_ERRORS)


async def _solve_one(
    case: dict, gateway: EvidenceGateway, contracts: Contracts, trace_path: Path
) -> dict:
    """Solve one case into its own trace file so a failed attempt leaves no partial events."""
    case_id = case["case_id"]
    trace_path.unlink(missing_ok=True)
    trace = TraceWriter(trace_path, contracts)
    trace.emit(case_id=case_id, event_type="case_received", actor="coordinator")
    output = await solve_case(case, gateway, trace)
    contracts.validate_output(output, f"outputs/{case_id}.json")
    if output.get("case_id") != case_id:
        raise ValueError(f"solver returned a mismatched case_id for {case_id}")
    trace.emit(case_id=case_id, event_type="case_finalized", actor="coordinator")
    return output


def _finalized_cases(trace_path: Path) -> set[str]:
    if not trace_path.exists():
        return set()
    finalized: set[str] = set()
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            if event.get("event_type") == "case_finalized":
                finalized.add(event["case_id"])
    return finalized


def _prepare_resume(output_root: Path, trace_path: Path) -> set[str]:
    """Keep only cases with both an output file and a finalized trace; drop the rest."""
    finalized = _finalized_cases(trace_path)
    done = {path.stem for path in output_root.glob("*.json")} & finalized
    for path in output_root.glob("*.json"):
        if path.stem not in done:
            path.unlink()
    if trace_path.exists():
        kept = [
            line
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["case_id"] in done
        ]
        trace_path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return done


async def _run(root: Path, *, resume: bool = False) -> None:
    settings = Settings.load(root)
    case_set = load_case_set(root)
    contracts = Contracts(root / "contracts" / "schemas")
    output_root = root / "outputs"
    trace_path = root / "traces" / "trace.jsonl"
    case_trace_path = trace_path.with_suffix(".case.tmp")
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if resume:
        done = _prepare_resume(output_root, trace_path)
        print(f"resume: {len(done)} cases already finalized", flush=True)
    else:
        for stale in output_root.glob("*.json"):
            stale.unlink()
        trace_path.unlink(missing_ok=True)
    trace_path.touch()

    pending = [case_id for case_id in case_set.case_ids if case_id not in done]
    failures = 0
    while pending:
        try:
            async with connect_gateway(
                settings.mcp_endpoint, settings.team_api_key, contracts
            ) as gateway:
                if not await gateway.list_tools():
                    raise RuntimeError("MCP Gateway returned no tools")
                while pending:
                    case_id = pending[0]
                    output = await _solve_one(
                        case_set.cases[case_id], gateway, contracts, case_trace_path
                    )
                    target = output_root / f"{case_id}.json"
                    temporary = target.with_suffix(".json.tmp")
                    temporary.write_text(
                        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    temporary.replace(target)
                    with trace_path.open("a", encoding="utf-8") as handle:
                        handle.write(case_trace_path.read_text(encoding="utf-8"))
                    pending.pop(0)
                    failures = 0
                    print(f"{case_id}: {output['assessment']['primary_issue']}", flush=True)
        except BaseException as exc:
            if not _is_transient(exc):
                raise
            failures += 1
            if failures >= MAX_CONNECTION_ATTEMPTS:
                raise RuntimeError(
                    f"MCP connection failed {failures} times in a row at {pending[0]}"
                ) from exc
            delay = min(2 ** (failures + 1), MAX_RETRY_DELAY_SECONDS)
            print(
                f"{pending[0]}: connection lost ({type(exc).__name__}), retry in {delay}s",
                flush=True,
            )
            await asyncio.sleep(delay)
    case_trace_path.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Day09 L3A student workflow")
    result.add_argument("--root", default=".", help="repository root (default: current directory)")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-inputs", help="validate case-set.json and all 100 inputs")
    commands.add_parser("mcp-tools", help="authenticate and list discovered MCP tools")
    run = commands.add_parser("run", help="run the implemented workflow for all cases")
    run.add_argument(
        "--resume",
        action="store_true",
        help="keep finalized cases from the previous run and solve only the missing ones",
    )
    commands.add_parser("validate", help="validate outputs and observable trace")
    package = commands.add_parser("package", help="validate and build the submission ZIP")
    package.add_argument("--output", default="dist/submission.zip")
    return result


def main() -> None:
    args = parser().parse_args()
    root = _root(args.root)
    try:
        if args.command == "validate-inputs":
            case_set = load_case_set(root)
            print(
                f"OK: {case_set.variant_id} / {case_set.version} / "
                f"{len(case_set.case_ids)} cases"
            )
        elif args.command == "mcp-tools":
            asyncio.run(_show_tools(root))
        elif args.command == "run":
            asyncio.run(_run(root, resume=args.resume))
        elif args.command == "validate":
            case_set = load_case_set(root)
            contracts = Contracts(root / "contracts" / "schemas")
            _, trace = validate_artifacts(root, case_set, contracts)
            print(f"OK: {len(case_set.case_ids)} outputs / {len(trace)} trace events")
        elif args.command == "package":
            destination = package_submission(root, root / args.output)
            print(f"OK: {destination}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
