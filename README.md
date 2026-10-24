# Eugene Plexus store

The store keeps what the gateway's OpenAI- and Anthropic-compatible doors
keep for people: uploaded files first (`/v1/files`, `/v1/uploads`), then
stored responses, batches and vector stores. Every install has one, on the
root by default, started and watched by the agent like the gateway and the
library.

Contract: [`openapi/store.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/store.yaml).
Design: [`docs/design/rows/row-53.md`](https://github.com/eugene-plexus/specs/blob/main/docs/design/rows/row-53.md).

## What it promises

- **Every object has one owner.** The gateway names the owner on every call
  from its own admission: the person the root verified, else the client key,
  else the owner's console. The store believes that name only beside a
  gateway's service token.
- **Only the owner sees it.** Another owner's id answers exactly as a missing
  one. The install owner sees counts and bytes per owner, never names or
  content.
- **A file is kept until it is deleted**, as OpenAI keeps one, unless its app
  asked for an end (one hour to 30 days) or the owner set a default life. An
  upload in progress lives one hour. The sweep deletes what ended: its row
  (SQLite's `secure_delete` is on) and its blob when no other file shares it.
- **A removed person's objects are hidden, then deleted**, as Workbench does
  (row 7). Each sweep asks the root, through this machine's agent, which
  people it still has. A person it no longer has is hidden at once and deleted
  30 days later (`removedPeopleDays`), or at once when the owner keeps a
  removed person's data "Not at all"; the root having them again first brings
  everything back. The owner's Delete now deletes at once, every copy.

## Where it keeps things

`<data>/store/store.sqlite3` and `<data>/store/blobs/`, each blob named by
its SHA-256. `<data>` is the folder of the store's config file, which under
the agent is Eugene's own data folder.

## Development

```bash
python -m venv .venv
. .venv/bin/activate            # or: .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install
pytest
python scripts/codegen.py       # regenerate models from the pinned specs
```

Licensed under Apache 2.0. Commits carry a DCO sign-off (`git commit -s`);
see [CONTRIBUTING.md](CONTRIBUTING.md).
