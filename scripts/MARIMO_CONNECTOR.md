# Marimo connector

`marimo_connector.py` is the project-local, dependency-free connector for
running code in an active Marimo kernel. It uses Marimo's HTTP API directly,
so it works from PowerShell even when the Bash pairing wrapper is unavailable.

The URL may be either the server URL or the URL copied from the Claude pairing
prompt. The connector removes a trailing `/--claude` suffix automatically.

## Usage

```powershell
python scripts/marimo_connector.py `
  --url "https://your-marimo-host.example" `
  --code "print('connected')"
```

For multiline code, use a file or stdin:

```powershell
python scripts/marimo_connector.py `
  --url "https://your-marimo-host.example" `
  --file .\check_marimo.py

Get-Content .\check_marimo.py -Raw |
  python scripts/marimo_connector.py --url "https://your-marimo-host.example"
```

If the server requires authentication, set the token in the environment:

```powershell
$env:MARIMO_TOKEN = "..."
python scripts/marimo_connector.py --url "https://your-marimo-host.example" --file .\check_marimo.py
```

The connector discovers the only active session automatically. If more than
one session is open, pass `--session SESSION_ID` explicitly. It streams
Marimo stdout/stderr and returns a non-zero exit code for connection, HTTP,
session-selection, or kernel-execution errors.

## Cell operations

Cell operations are committed through `marimo._code_mode`; the connector never
edits the notebook `.py` file directly. Cell references may be an ID, name, or
zero-based index.

```powershell
# Metadata only; does not print cell source.
python scripts/marimo_connector.py --url $url --list-cells

# Read the current source of one cell.
python scripts/marimo_connector.py --url $url --read-cell imports

# Create and run a visible cell.
python scripts/marimo_connector.py --url $url --create-cell `
  --name connector_check --show-code --run-after --file .\check.py

# Edit and optionally run an existing cell.
python scripts/marimo_connector.py --url $url --edit-cell connector_check `
  --run-after --file .\check.py

# Run an existing cell without editing it.
python scripts/marimo_connector.py --url $url --run-cell connector_check
```

Deletion is guarded and requires `--confirm-delete`. To remove a notebook
suffix while preserving its boundary cell, use:

```powershell
python scripts/marimo_connector.py --url $url --delete-after monitor --confirm-delete
```

Moving a cell requires exactly one of `--before REF` or `--after REF`.
