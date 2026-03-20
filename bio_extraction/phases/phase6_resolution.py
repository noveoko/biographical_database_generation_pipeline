# REQUIRES: python-Levenshtein

from __future__ import annotations

import json
import sqlite3
import unicodedata
from datetime import datetime
from typing import List, Optional, Tuple

import Levenshtein

from bio_extraction.config import get_settings
from bio_extraction.contracts import ExtractionResult, ResolutionResult, ResolvedPerson, PersonEntity
from bio_extraction.exceptions import ResolutionError
from bio_extraction.logging_config import get_phase_logger
from bio_extraction.protocol import PhaseProtocol


class ResolutionPhase(PhaseProtocol[ExtractionResult, ResolutionResult]):
    """Phase 6 — Entity Resolution & Storage."""

    @property
    def phase_name(self) -> str:
        return "phase6_resolution"

    def __init__(self) -> None:
        self.logger = get_phase_logger(self.phase_name)
        self.config = get_settings()
        self.conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite database and schema."""
        try:
            conn = sqlite3.connect(self.config.resolution.sqlite_path)
            cursor = conn.cursor()

            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    surname TEXT NOT NULL,
                    given_names TEXT NOT NULL,
                    surname_normalized TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    event_date TEXT,
                    location TEXT,
                    detail TEXT,
                    confidence REAL NOT NULL,
                    extraction_method TEXT NOT NULL,
                    source_doc_id TEXT NOT NULL,
                    source_slice_id TEXT NOT NULL,
                    raw_text TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(person_id, event_type, event_date, location, detail)
                );

                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL UNIQUE,
                    filename TEXT,
                    source_url TEXT,
                    warc_id TEXT,
                    processed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_persons_surname_norm ON persons(surname_normalized);
                CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
                """
            )

            conn.commit()
            return conn
        except Exception as e:
            raise ResolutionError(self.phase_name, "N/A", f"DB init failed: {e}")

    def run(self, input_data: ExtractionResult) -> ResolutionResult:
        """Resolve entities and persist them."""
        resolved: List[ResolvedPerson] = []

        try:
            for entity in input_data.entities:
                person_id, is_new, score = self._resolve_person(entity)

                self._insert_events(person_id, entity, input_data.doc_id)

                resolved.append(
                    ResolvedPerson(
                        person_db_id=person_id,
                        entity_id=entity.entity_id,
                        is_new=is_new,
                        merge_confidence=score if not is_new else None,
                    )
                )

            self._insert_source(input_data.doc_id)

            if not resolved:
                raise ResolutionError(self.phase_name, input_data.doc_id, "All entities failed")

            return ResolutionResult(
                doc_id=input_data.doc_id,
                resolved_persons=resolved,
                resolved_at=datetime.utcnow(),
            )

        except ResolutionError:
            raise
        except Exception as e:
            raise ResolutionError(self.phase_name, input_data.doc_id, str(e))

    # ---------------- NORMALIZATION ----------------

    def _normalize_surname(self, surname: str) -> str:
        """Normalize surname: lowercase + strip Polish diacritics."""
        mapping = str.maketrans("ąćęłńóśźż", "acelnoszz")
        return surname.lower().translate(mapping).strip()

    def _normalize_date(self, date: Optional[str]) -> Optional[str]:
        """Normalize date into ISO or year."""
        if not date:
            return None

        date = date.strip().lower()

        import re

        # full date dd.mm.yyyy
        m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", date)
        if m:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

        # year only
        m = re.search(r"(\d{4})", date)
        if m:
            return m.group(1)

        return None

    # ---------------- MATCHING ----------------

    def _resolve_person(self, entity: PersonEntity) -> Tuple[int, bool, Optional[float]]:
        """Find or create person."""
        norm_surname = self._normalize_surname(entity.surname)
        cursor = self.conn.cursor()

        # exact match
        cursor.execute(
            "SELECT id, given_names FROM persons WHERE surname_normalized = ?",
            (norm_surname,),
        )
        rows = cursor.fetchall()

        if rows:
            best_id = self._pick_best_given_name_match(rows, entity.given_names)
            self._update_timestamp(best_id)
            return best_id, False, 1.0

        # fuzzy match
        cursor.execute("SELECT id, surname_normalized, given_names FROM persons")
        candidates = cursor.fetchall()

        best_score = 0.0
        best_id = None

        for pid, s_norm, g_names_json in candidates:
            score = Levenshtein.ratio(norm_surname, s_norm)
            if score >= self.config.resolution.fuzzy_match_threshold:
                if score > best_score:
                    best_score = score
                    best_id = pid

        if best_id:
            self._update_timestamp(best_id)
            return best_id, False, best_score

        # create new
        now = datetime.utcnow().isoformat()
        cursor.execute(
            """
            INSERT INTO persons (surname, given_names, surname_normalized, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                entity.surname,
                json.dumps(entity.given_names),
                norm_surname,
                now,
                now,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid, True, None

    def _pick_best_given_name_match(self, rows, given_names: List[str]) -> int:
        """Pick best candidate based on Jaccard similarity."""
        target = set(given_names)
        best_score = -1
        best_id = rows[0][0]

        for pid, g_names_json in rows:
            existing = set(json.loads(g_names_json))
            union = len(target | existing)
            inter = len(target & existing)
            score = inter / union if union else 0

            if score > best_score:
                best_score = score
                best_id = pid

        return best_id

    def _update_timestamp(self, person_id: int) -> None:
        """Update updated_at timestamp."""
        self.conn.execute(
            "UPDATE persons SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), person_id),
        )
        self.conn.commit()

    # ---------------- EVENTS ----------------

    def _insert_events(self, person_id: int, entity: PersonEntity, doc_id: str) -> None:
        """Insert events for entity."""
        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()

        def safe_insert(event_type, date=None, location=None, detail=None):
            try:
                cursor.execute(
                    """
                    INSERT INTO events (
                        person_id, event_type, event_date, location, detail,
                        confidence, extraction_method, source_doc_id, source_slice_id,
                        raw_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person_id,
                        event_type,
                        date,
                        location,
                        detail,
                        entity.confidence,
                        entity.extraction_method.value,
                        doc_id,
                        entity.slice_id,
                        entity.raw_text,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                self.logger.warning(f"Duplicate event skipped for person {person_id}")

        # birth / death
        if entity.birth_date:
            safe_insert("birth", self._normalize_date(entity.birth_date))

        if entity.death_date:
            safe_insert("death", self._normalize_date(entity.death_date))

        # locations
        for loc in entity.locations:
            safe_insert("residence", location=loc)

        # roles
        for role in entity.roles:
            safe_insert("occupation", detail=role)

        self.conn.commit()

    # ---------------- SOURCE ----------------

    def _insert_source(self, doc_id: str) -> None:
        """Insert source record."""
        try:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO sources (doc_id, filename, source_url, warc_id, processed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, None, None, None, datetime.utcnow().isoformat()),
            )
            self.conn.commit()
        except Exception as e:
            self.logger.warning(f"Failed to insert source: {e}")


if __name__ == "__main__":
    from bio_extraction.contracts import ExtractionMethod

    phase = ResolutionPhase()

    entity = PersonEntity(
        entity_id="e1",
        slice_id="s1",
        surname="Kowalski",
        given_names=["Jan"],
        birth_date="12.03.1892",
        death_date=None,
        locations=["Warszawa"],
        roles=["lekarz"],
        raw_text="Jan Kowalski ur. 12.03.1892 lekarz Warszawa",
        confidence=0.95,
        extraction_method=ExtractionMethod.REGEX_CACHED,
    )

    result = phase.run(
        ExtractionResult(
            doc_id="doc_test",
            entities=[entity],
            extracted_at=datetime.utcnow(),
        )
    )

    print(result.model_dump())