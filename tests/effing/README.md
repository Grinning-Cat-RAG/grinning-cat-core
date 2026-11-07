# `tests/effing/` — EffING integration tests (what stays here and why)

The Efficient Ingestion plugin (`cat-efficient-ingestion`) owns its own
standalone, import-safe test suite in the plugin repository. This directory is
the **deliberate cross-repo integration exception**: it holds only the tests
that genuinely need the full `CheshireCat` / `BillTheLizard` boot, the real
FastAPI routes, the real hook dispatcher, and the real Redis/Qdrant/file-manager
harness.

Plugin-owned suites (moved out of this directory; run from the plugin repo):

- `test_ingestion_status_phases.py` — registry + `completed_phases` diary
  (pure Redis CRUD).
- `test_final_wave_findings.py` — download lease, NX create, resume lease,
  batch heartbeat (pure with fake cats).
- `test_known_blockers.py` — B1 status listing, B3/B3b download resumability
  (pure registry + CRUD).
- `test_round2_findings.py` — NN1-NN3 heartbeat/lease (pure with fake cats).
- `test_plugin_dependency_metadata.py` — manifest + pyproject self-consistency.

They share one import-safe harness at
`cat-efficient-ingestion/_testkit.py` (`bootstrap` / `teardown` / `run`). The
harness forces `CAT_REDIS_DB=1` before importing `cat`, aborts if the resolved
Redis db is not the dedicated test db, and only ever deletes the
`agents:<id>:*` keys a test created.

## What stays here (and why the full host is required)

| File | Why it needs the real host |
|---|---|
| `test_efficient_ingestion_installation.py` | `BillTheLizard` startup validation, system-only plugin contract. |
| `test_efficient_ingestion.py` | `ServiceFactory` resolution and the real `/ingestion/settings` routes. |
| `test_store_hook_once.py` | Real `PluginManager` dispatcher + real engine on a real cat. |
| `test_url_source_persistence.py` | Real `CheshireCat`, `LocalFileManager` and vector store for URL byte persistence. |
| `test_final_wave_findings.py` | The chunker-change case through the real `PUT /chunking/settings` route. |
| `test_known_blockers.py` | B2/B2b (real `CheshireCat` settings) and B4 (real dispatcher via `StrayCat`). |
| `test_round2_findings.py` | Concurrent two-agent chunker isolation (two real `CheshireCat` instances + real chunkers). |
| `test_plugin_dependency_metadata.py` | Host-side loader (`Plugin.missing_dependencies`, `lizard` load) and the `min_cat_version` pin. |

## Running

From the core checkout, against the integration stack (Redis with RedisJSON +
Qdrant):

```bash
CAT_REDIS_HOST=<redis ip> CAT_QDRANT_HOST=<qdrant ip> .venv/bin/python -m pytest tests/effing -q
```

`tests/conftest.py` forces `CAT_REDIS_DB=1` before the `Database` singleton is
created. Do not add plugin-only behaviour tests here: add them as a standalone
script in the plugin instead.
