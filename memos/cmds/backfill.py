"""Backfill missing plugin metadata for historical entities.

Reprocesses image entities that lack a given plugin's metadata key — e.g. old
screenshots captured under a previous VLM model, which therefore never got the
current model's `structured_vlm_v1_<model>` key.

Design notes:
- Reuses the plugin's own predict + key functions, so the value written and the
  key it lands under stay in lockstep with the live pipeline.
- Runs at a chosen concurrency in the CLI process, independent of the server's
  per-plugin semaphore (so the live `watch` worker keeps its own capacity).
- Writes each result back through PATCH /api/entities/{id}/metadata, which also
  refreshes the FTS + vector index for that entity in the same pass.
- Idempotent and resumable: enumeration only returns still-missing entities and
  pages forward by id, so re-running picks up wherever it left off.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

import httpx
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from memos import crud
from memos.config import settings
from memos.schemas import MetadataType

# The save step (PATCH) re-embeds + reindexes synchronously, so it can briefly
# fail under concurrency. Retry it to avoid throwing away a completed VLM call.
_PATCH_RETRIES = 3
# DB read pages must survive a transient Postgres connection drop on a long run.
_DB_RETRIES = 5


@dataclass
class Backfiller:
    name: str  # metadata `source` label written back
    key: str  # target metadata key to fill
    data_type: str  # MetadataType value for the PATCH payload
    predict: Callable[[str, int], Awaitable[Optional[str]]]  # (filepath, id) -> value | None


def _structured_vlm_backfiller() -> Backfiller:
    from memos.plugins.structured_vlm.main import (
        metadata_field_name,
        predict_structured,
    )

    v = settings.vlm

    async def predict(filepath: str, entity_id: int) -> Optional[str]:
        result = await predict_structured(
            endpoint=v.endpoint,
            modelname=v.modelname,
            img_path=filepath,
            token=v.token,
            max_tokens=v.max_tokens,
            disable_thinking=v.disable_thinking,
            entity_id=entity_id,
        )
        return result.model_dump_json() if result is not None else None

    return Backfiller(
        name="structured_vlm",
        key=metadata_field_name(v.modelname),
        data_type=MetadataType.JSON_DATA.value,
        predict=predict,
    )


# Registry of supported plugins. Add an entry to extend (e.g. plain "vlm").
BACKFILLERS = {
    "structured_vlm": _structured_vlm_backfiller,
}


def get_backfiller(plugin: str) -> Backfiller:
    if plugin not in BACKFILLERS:
        supported = ", ".join(sorted(BACKFILLERS))
        raise ValueError(
            f"Unsupported backfill plugin '{plugin}'. Supported: {supported}"
        )
    return BACKFILLERS[plugin]()


async def _process_one(
    client: httpx.AsyncClient,
    bf: Backfiller,
    semaphore: asyncio.Semaphore,
    counters: dict,
    pbar: tqdm,
    entity_id: int,
    filepath: str,
) -> None:
    try:
        async with semaphore:
            try:
                value = await bf.predict(filepath, entity_id)
            except Exception as e:  # predict should swallow its own errors; guard anyway
                counters["failed"] += 1
                tqdm.write(f"[fail] entity={entity_id} predict error: {e}")
                return
        if value is None:
            # predict_* already logged a `category=...` line for triage.
            counters["failed"] += 1
            return
        # The VLM call already succeeded — the result is expensive, the local
        # PATCH (which synchronously re-embeds + reindexes) is cheap to retry. So
        # retry it a few times with backoff rather than discarding the VLM work
        # when the server is briefly overloaded.
        url = f"{settings.server_endpoint}/api/entities/{entity_id}/metadata"
        payload = {
            "metadata_entries": [
                {
                    "key": bf.key,
                    "value": value,
                    "source": bf.name,
                    "data_type": bf.data_type,
                }
            ]
        }
        last_err = None
        for attempt in range(_PATCH_RETRIES):
            try:
                resp = await client.patch(url, json=payload, timeout=120)
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    counters["done"] += 1
                    return
                last_err = f"http {resp.status_code}: {resp.text[:120]}"
                if not 500 <= resp.status_code < 600:
                    break  # 4xx is terminal — retrying won't help
            if attempt < _PATCH_RETRIES - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
        counters["failed"] += 1
        tqdm.write(f"[fail] entity={entity_id} patch: {last_err}")
    finally:
        pbar.update(1)


async def _run(
    bf: Backfiller,
    library_ids: Optional[List[int]],
    concurrency: int,
    limit: Optional[int],
    page_size: int,
    SessionLocal,
) -> None:
    with SessionLocal() as db:
        total = crud.count_entities_missing_metadata(
            db=db, key=bf.key, library_ids=library_ids
        )
    if limit is not None:
        total = min(total, limit)
    print(
        f"Backfilling '{bf.name}' key={bf.key}: "
        f"{total} entities to process, concurrency={concurrency}"
    )
    if total == 0:
        return

    semaphore = asyncio.Semaphore(concurrency)
    counters = {"done": 0, "failed": 0}
    processed = 0
    after_id = 0
    pbar = tqdm(total=total, unit="img", smoothing=0.05)
    # Size the pool to the concurrency so PATCHes don't thrash a too-small
    # keepalive pool (the default keepalive of 20 < typical concurrency).
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency
    )
    try:
        async with httpx.AsyncClient(limits=limits) as client:
            while True:
                remaining = None if limit is None else limit - processed
                if remaining is not None and remaining <= 0:
                    break
                fetch_n = page_size if remaining is None else min(page_size, remaining)
                # A transient Postgres blip ("server closed the connection")
                # must not kill a multi-day run — retry the page fetch, letting
                # the pool's pre-ping recycle the dead connection.
                rows = None
                for attempt in range(_DB_RETRIES):
                    try:
                        with SessionLocal() as db:
                            rows = crud.list_entities_missing_metadata(
                                db=db,
                                key=bf.key,
                                library_ids=library_ids,
                                after_id=after_id,
                                limit=fetch_n,
                            )
                        break
                    except (OperationalError, DBAPIError) as e:
                        if attempt == _DB_RETRIES - 1:
                            raise
                        delay = 2 * (attempt + 1)
                        tqdm.write(
                            f"[db] page fetch failed (retry {attempt + 1}/{_DB_RETRIES} "
                            f"in {delay}s): {type(e).__name__}"
                        )
                        await asyncio.sleep(delay)
                if not rows:
                    break
                after_id = rows[-1][0]
                processed += len(rows)
                await asyncio.gather(
                    *(
                        _process_one(
                            client, bf, semaphore, counters, pbar, eid, fp
                        )
                        for eid, fp in rows
                    )
                )
                pbar.set_postfix(done=counters["done"], failed=counters["failed"])
    finally:
        pbar.close()
    print(
        f"Done. patched={counters['done']} failed/skipped={counters['failed']}. "
        f"Re-run the same command to retry skipped entities."
    )


def run_backfill(
    plugin: str = "structured_vlm",
    library_id: Optional[int] = None,
    concurrency: Optional[int] = None,
    limit: Optional[int] = None,
    page_size: int = 2000,
    dry_run: bool = False,
) -> None:
    bf = get_backfiller(plugin)
    library_ids = [library_id] if library_id is not None else None
    if concurrency is None:
        concurrency = settings.vlm.concurrency

    # pool_pre_ping transparently discards connections the server has closed
    # (idle timeout, restart) instead of handing back a dead one.
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)
    try:
        if dry_run:
            with SessionLocal() as db:
                total = crud.count_entities_missing_metadata(
                    db=db, key=bf.key, library_ids=library_ids
                )
            cap = f" (would process at most {limit})" if limit else ""
            print(f"[dry-run] '{bf.name}' key={bf.key}: {total} entities missing{cap}")
            return
        try:
            asyncio.run(_run(bf, library_ids, concurrency, limit, page_size, SessionLocal))
        except KeyboardInterrupt:
            print("\nInterrupted. Progress is saved; re-run to resume.")
    finally:
        engine.dispose()
