"""Tests for plugin-metadata backfill enumeration (crud) and key derivation.

The backfill command reuses the structured_vlm plugin's own key function, so a
mismatch here would silently reprocess everything or nothing.
"""
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memos.models import Base, EntityModel, EntityMetadataModel
from memos.schemas import MetadataSource, MetadataType
from memos import crud
from memos.plugins.structured_vlm.main import metadata_field_name


KEY = "structured_vlm_v1_qwen3_6_35b"


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add_entity(db, *, library_id=1, file_type_group="image"):
    e = EntityModel(
        filepath=f"/fake/{id(db)}-{library_id}-{db.query(EntityModel).count()}.webp",
        filename="x.webp",
        size=1,
        file_created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        file_last_modified_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        file_type="webp",
        file_type_group=file_type_group,
        library_id=library_id,
        folder_id=1,
    )
    db.add(e)
    db.commit()
    return e


def _add_metadata(db, entity, key, value):
    db.add(
        EntityMetadataModel(
            entity_id=entity.id,
            key=key,
            value=value,
            source_type=MetadataSource.PLUGIN_GENERATED,
            source="structured_vlm",
            data_type=MetadataType.JSON_DATA,
        )
    )
    db.commit()


def _missing_ids(db, **kw):
    return [eid for eid, _ in crud.list_entities_missing_metadata(db=db, key=KEY, **kw)]


def test_entity_without_the_key_is_missing():
    db = _make_session()
    e = _add_entity(db)
    assert _missing_ids(db) == [e.id]
    assert crud.count_entities_missing_metadata(db=db, key=KEY) == 1


def test_entity_with_nonempty_value_is_not_missing():
    db = _make_session()
    e = _add_entity(db)
    _add_metadata(db, e, KEY, '{"extractor":"x","primary":{}}')
    assert _missing_ids(db) == []
    assert crud.count_entities_missing_metadata(db=db, key=KEY) == 0


def test_empty_and_brace_values_count_as_missing():
    db = _make_session()
    e_empty = _add_entity(db)
    _add_metadata(db, e_empty, KEY, "")
    e_brace = _add_entity(db)
    _add_metadata(db, e_brace, KEY, "{}")
    assert set(_missing_ids(db)) == {e_empty.id, e_brace.id}


def test_other_model_key_does_not_satisfy_current_key():
    db = _make_session()
    e = _add_entity(db)
    _add_metadata(db, e, "minicpm_v_result", "old description")
    # has an old model's result but not the current structured key -> still missing
    assert _missing_ids(db) == [e.id]


def test_non_image_entities_are_skipped():
    db = _make_session()
    _add_entity(db, file_type_group="document")
    img = _add_entity(db, file_type_group="image")
    assert _missing_ids(db) == [img.id]


def test_library_filter():
    db = _make_session()
    a = _add_entity(db, library_id=1)
    b = _add_entity(db, library_id=2)
    assert _missing_ids(db, library_ids=[1]) == [a.id]
    assert set(_missing_ids(db, library_ids=[1, 2])) == {a.id, b.id}


def test_keyset_pagination_with_after_id_and_limit():
    db = _make_session()
    ids = [_add_entity(db).id for _ in range(5)]
    first = crud.list_entities_missing_metadata(db=db, key=KEY, limit=2)
    assert [eid for eid, _ in first] == ids[:2]
    nxt = crud.list_entities_missing_metadata(
        db=db, key=KEY, after_id=first[-1][0], limit=2
    )
    assert [eid for eid, _ in nxt] == ids[2:4]


def test_returns_filepath_for_processing():
    db = _make_session()
    e = _add_entity(db)
    rows = crud.list_entities_missing_metadata(db=db, key=KEY)
    assert rows == [(e.id, e.filepath)]


def test_metadata_field_name_matches_expected_key():
    # Ties the backfill target to the plugin's own derivation. If the plugin's
    # key scheme changes, this fails loudly instead of backfilling a stale key.
    assert metadata_field_name("qwen3.6-35b") == KEY
