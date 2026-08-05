#!/usr/bin/env python3
r"""Run code in an active Marimo kernel over its HTTP API.

This is a small, dependency-free connector for environments where the
Marimo pairing shell script is unavailable. It intentionally talks only to
Marimo's documented session and kernel-execute endpoints; it does not start
or modify a notebook server.

Examples:
    python scripts/marimo_connector.py --url https://example.run \
        --code "print('connected')"

    Get-Content .\check.py | python scripts/marimo_connector.py \
        --url https://example.run

Authentication:
    Set MARIMO_TOKEN in the environment. The token is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT = 120
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


class MarimoConnectorError(RuntimeError):
    """An actionable connection or Marimo API error."""


def normalize_url(raw_url: str) -> str:
    """Normalize a Marimo URL, including the ``/--claude`` prompt suffix."""

    url = raw_url.strip()
    if not url:
        raise MarimoConnectorError("Marimo URL is empty")

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MarimoConnectorError(
            "Marimo URL must include http:// or https:// and a hostname"
        )

    path = parsed.path.rstrip("/")
    if path == "/--claude":
        path = ""
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def auth_headers(token: str | None) -> dict[str, str]:
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
        **(headers or {}),
    }
    data = None
    if body is not None:
        request_headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")

    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        suffix = f": {detail[:500]}" if detail else ""
        raise MarimoConnectorError(
            f"Marimo returned HTTP {exc.code} for {method} {url}{suffix}"
        ) from exc
    except URLError as exc:
        raise MarimoConnectorError(f"Could not reach Marimo at {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise MarimoConnectorError(f"Timed out reaching Marimo at {url}") from exc
    except json.JSONDecodeError as exc:
        raise MarimoConnectorError(f"Marimo returned invalid JSON from {url}") from exc


def select_session(
    base_url: str,
    requested_session: str | None,
    *,
    headers: dict[str, str],
    timeout: int,
) -> str:
    if requested_session:
        return requested_session

    sessions = request_json(
        f"{base_url}/api/sessions", headers=headers, timeout=timeout
    )
    if not isinstance(sessions, dict) or not sessions:
        raise MarimoConnectorError(
            "No active Marimo sessions found. Open the notebook in its UI first."
        )

    session_ids = list(sessions)
    if len(session_ids) > 1:
        details = []
        for session_id in session_ids:
            metadata = sessions.get(session_id) or {}
            filename = metadata.get("filename") or metadata.get("path") or ""
            details.append(f"  {session_id} {filename}".rstrip())
        raise MarimoConnectorError(
            "Multiple active Marimo sessions found; pass --session explicitly:\n"
            + "\n".join(details)
        )
    return session_ids[0]


def read_code(args: argparse.Namespace) -> str:
    if args.code is not None:
        return args.code
    if args.file is not None:
        try:
            return Path(args.file).read_text(encoding="utf-8")
        except OSError as exc:
            raise MarimoConnectorError(f"Could not read code file {args.file}: {exc}") from exc
    if sys.stdin.isatty():
        raise MarimoConnectorError("Provide --code, --file, or code through stdin")
    code = sys.stdin.read()
    if not code.strip():
        raise MarimoConnectorError("Code from stdin is empty")
    return code


def execute_code(
    base_url: str,
    session_id: str,
    code: str,
    *,
    headers: dict[str, str],
    timeout: int,
) -> int:
    request_headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Marimo-Session-Id": session_id,
        "User-Agent": DEFAULT_USER_AGENT,
        **headers,
    }
    request = Request(
        f"{base_url}/api/kernel/execute",
        data=json.dumps({"code": code}).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )

    current_event = ""
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                payload = line[len("data:") :].strip()
                try:
                    message = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise MarimoConnectorError(
                        f"Marimo sent malformed SSE data for event {current_event!r}"
                    ) from exc

                if current_event == "stdout":
                    sys.stdout.write(str(message.get("data", "")))
                    sys.stdout.flush()
                elif current_event == "stderr":
                    sys.stderr.write(str(message.get("data", "")))
                    sys.stderr.flush()
                elif current_event == "done":
                    if message.get("success") is False:
                        error = message.get("error") or {}
                        detail = error.get("msg") or error or "unknown Marimo error"
                        raise MarimoConnectorError(str(detail))
                    output = (message.get("output") or {}).get("data")
                    if output:
                        sys.stdout.write(str(output))
                        sys.stdout.flush()
                    return 0
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        suffix = f": {detail[:500]}" if detail else ""
        raise MarimoConnectorError(
            f"Marimo returned HTTP {exc.code} while executing code{suffix}"
        ) from exc
    except URLError as exc:
        raise MarimoConnectorError(f"Could not execute code in Marimo: {exc.reason}") from exc
    except TimeoutError as exc:
        raise MarimoConnectorError("Timed out while executing code in Marimo") from exc

    raise MarimoConnectorError("Marimo closed the stream without a done event")


def _quoted(value: str) -> str:
    """Return a safely embedded Python string literal."""

    return json.dumps(value)


def build_cell_operation(args: argparse.Namespace, code: str | None) -> str:
    """Build scratchpad code that uses marimo._code_mode for cell operations."""

    target = (
        args.read_cell
        or args.edit_cell
        or args.run_cell
        or args.delete_cell
        or args.delete_after
        or args.move_cell
    )
    target_literal = _quoted(target) if target is not None else "None"
    run_after = bool(args.run_after)

    resolve = textwrap.dedent(
        f"""
        _target = {target_literal}
        _cells = list(_ctx.cells)
        _resolved = None
        _anchor_index = None
        for _index, _cell in enumerate(_cells):
            _cell_id = str(_cell.id)
            if _target in {{_cell_id, str(_cell.name), str(_index)}}:
                _resolved = _cell_id
                _anchor_index = _index
                break
        if _resolved is None:
            raise KeyError(f"Cell reference not found: {{_target}}")
        """
    ).strip()

    if args.list_cells:
        body = """
        _rows = []
        for _index, _cell in enumerate(list(_ctx.cells)):
            _rows.append({
                "index": _index,
                "id": str(_cell.id),
                "name": str(_cell.name),
                "status": str(_cell.status),
                "code_length": len(_cell.code),
                "error_count": len(_cell.errors),
            })
        print(_json.dumps(_rows, ensure_ascii=False, default=str))
        """
    elif args.read_cell:
        body = resolve + "\n" + textwrap.dedent("""
        _cell = next(_cell for _cell in _cells if str(_cell.id) == _resolved)
        print(_json.dumps({
            "index": next(_index for _index, _item in enumerate(_cells) if str(_item.id) == _resolved),
            "id": str(_cell.id),
            "name": str(_cell.name),
            "status": str(_cell.status),
            "code": _cell.code,
            "errors": [str(_error) for _error in _cell.errors],
        }, ensure_ascii=False, default=str))
        """).strip()
    elif args.create_cell:
        assert code is not None
        before = _quoted(args.before) if args.before else "None"
        after = _quoted(args.after) if args.after else "None"
        name = _quoted(args.name) if args.name else "None"
        body = f"""
        _new_id = _ctx.create_cell(
            {_quoted(code)},
            before={before},
            after={after},
            hide_code={not args.show_code!r},
            disabled={bool(args.disabled)!r},
            name={name},
        )
        print(_json.dumps({{"created_id": str(_new_id)}}, default=str))
        {"_ctx.run_cell(_new_id)" if run_after else ""}
        """
    elif args.edit_cell:
        assert code is not None
        body = resolve + "\n" + textwrap.dedent(f"""
        _ctx.edit_cell(
            _resolved,
            code={_quoted(code)},
            hide_code={None if args.show_code is None else not args.show_code!r},
            disabled={args.disabled if args.disabled is not None else None!r},
            name={_quoted(args.name) if args.name else None!r},
        )
        print(_json.dumps({{"edited_id": _resolved}}, default=str))
        {"_ctx.run_cell(_resolved)" if run_after else ""}
        """).strip()
    elif args.run_cell:
        body = resolve + "\n" + textwrap.dedent("""
        _ctx.run_cell(_resolved)
        print(_json.dumps({"run_id": _resolved}, default=str))
        """).strip()
    elif args.delete_cell:
        if not args.confirm_delete:
            raise MarimoConnectorError(
                "Deleting a cell requires --confirm-delete"
            )
        body = resolve + "\n" + textwrap.dedent("""
        _ctx.delete_cell(_resolved)
        print(_json.dumps({"deleted_id": _resolved}, default=str))
        """).strip()
    elif args.delete_after:
        if not args.confirm_delete:
            raise MarimoConnectorError(
                "Deleting cells requires --confirm-delete"
            )
        body = resolve + "\n" + textwrap.dedent("""
        _delete_ids = [
            str(_cell.id)
            for _index, _cell in enumerate(_cells)
            if _index > _anchor_index
        ]
        for _cell_id in reversed(_delete_ids):
            _ctx.delete_cell(_cell_id)
        print(_json.dumps({"deleted_ids": _delete_ids, "count": len(_delete_ids)}, default=str))
        """).strip()
    elif args.move_cell:
        if bool(args.before) == bool(args.after):
            raise MarimoConnectorError(
                "Moving a cell requires exactly one of --before or --after"
            )
        body = resolve + "\n" + textwrap.dedent(f"""
        _ctx.move_cell(
            _resolved,
            before={_quoted(args.before) if args.before else None!r},
            after={_quoted(args.after) if args.after else None!r},
        )
        print(_json.dumps({{"moved_id": _resolved}}, default=str))
        """).strip()
    else:
        raise MarimoConnectorError("No Marimo operation selected")

    body = textwrap.dedent(body).strip()
    return (
        "import json as _json\n"
        "import marimo._code_mode as _cm\n\n"
        "async with _cm.get_context() as _ctx:\n"
        + textwrap.indent(body, "    ")
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Marimo server URL")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list-cells", action="store_true", help="List cell metadata")
    action.add_argument("--read-cell", metavar="REF", help="Read a cell by ID, name, or index")
    action.add_argument("--create-cell", action="store_true", help="Create a durable cell")
    action.add_argument("--edit-cell", metavar="REF", help="Edit a cell by ID, name, or index")
    action.add_argument("--run-cell", metavar="REF", help="Run a cell by ID, name, or index")
    action.add_argument("--delete-cell", metavar="REF", help="Delete a cell (requires confirmation)")
    action.add_argument("--delete-after", metavar="REF", help="Delete every cell after REF (requires confirmation)")
    action.add_argument("--move-cell", metavar="REF", help="Move a cell by ID, name, or index")
    code_group = parser.add_mutually_exclusive_group()
    code_group.add_argument("--code", help="Python code to execute")
    code_group.add_argument("--file", help="Read Python code from this file")
    parser.add_argument("--name", help="Cell name for --create-cell or --edit-cell")
    parser.add_argument("--before", metavar="REF", help="Insert/move before this cell")
    parser.add_argument("--after", metavar="REF", help="Insert/move after this cell")
    parser.add_argument("--show-code", action="store_true", default=None, help="Show created/edited cell code")
    parser.add_argument("--disabled", action="store_true", default=None, help="Disable a created/edited cell")
    parser.add_argument("--run-after", action="store_true", help="Run a created or edited cell")
    parser.add_argument("--confirm-delete", action="store_true", help="Confirm destructive cell deletion")
    parser.add_argument("--session", help="Explicit Marimo session ID")
    parser.add_argument(
        "--token",
        help="Bearer token; prefer the MARIMO_TOKEN environment variable",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2

    try:
        base_url = normalize_url(args.url)
        operation_selected = any(
            getattr(args, name)
            for name in (
                "list_cells",
                "read_cell",
                "create_cell",
                "edit_cell",
                "run_cell",
                "delete_cell",
                "delete_after",
                "move_cell",
            )
        )
        if operation_selected:
            if (
                args.list_cells
                or args.read_cell
                or args.run_cell
                or args.delete_cell
                or args.delete_after
                or args.move_cell
            ):
                if args.code is not None or args.file is not None:
                    raise MarimoConnectorError(
                        "This operation does not accept --code or --file"
                    )
                code = None
            else:
                code = read_code(args)
            code_to_execute = build_cell_operation(args, code)
        else:
            if any(
                value is not None
                for value in (args.name, args.before, args.after, args.run_after, args.confirm_delete)
            ):
                raise MarimoConnectorError(
                    "Cell-only flags require a cell operation such as --create-cell"
                )
            code_to_execute = read_code(args)
        token = args.token or os.environ.get("MARIMO_TOKEN")
        headers = auth_headers(token)
        session_id = select_session(
            base_url, args.session, headers=headers, timeout=args.timeout
        )
        return execute_code(
            base_url,
            session_id,
            code_to_execute,
            headers=headers,
            timeout=args.timeout,
        )
    except MarimoConnectorError as exc:
        print(f"marimo-connector: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
