from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}
MAX_DOCUMENT_BYTES = 15 * 1024 * 1024
GID_PATTERN = re.compile(r"\b\d{12,20}\b")


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    extension: str
    size_bytes: int
    sha256: str
    added_at: str
    text_chars: int
    detected_gids: list[str]


class CaseDocumentStore:
    """Local-only store for user-supplied investigation documents."""

    def __init__(self, root: str | Path = "case_documents") -> None:
        self.root = Path(root)

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(str(filename)).name.strip()
        name = re.sub(r"[^\w.()\- ]+", "_", name, flags=re.UNICODE).strip(" .")
        if not name:
            raise ValueError("У документа отсутствует допустимое имя")
        return name[:180]

    @staticmethod
    def _decode_text(content: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")

    @classmethod
    def _extract_text(cls, extension: str, content: bytes) -> str:
        if extension in {".txt", ".md", ".csv", ".json"}:
            return cls._decode_text(content)
        if extension == ".docx":
            from docx import Document

            document = Document(io.BytesIO(content))
            blocks = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
            return "\n".join(blocks)
        if extension == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        raise ValueError(f"Неподдерживаемый формат: {extension}")

    def add_document(
        self, filename: str, content: bytes, *, known_gids: set[str] | None = None
    ) -> tuple[DocumentRecord, bool]:
        safe_name = self._safe_filename(filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Формат {extension or 'без расширения'} не поддерживается")
        if not content:
            raise ValueError("Документ пуст")
        if len(content) > MAX_DOCUMENT_BYTES:
            raise ValueError("Размер документа превышает 15 MB")

        digest = hashlib.sha256(content).hexdigest()
        document_id = digest[:16]
        metadata_path = self.root / f"{document_id}.meta.json"
        if metadata_path.exists():
            return DocumentRecord(**json.loads(metadata_path.read_text(encoding="utf-8"))), False

        try:
            text = self._extract_text(extension, content).strip()
        except Exception as exc:
            raise ValueError(f"Не удалось извлечь текст из {safe_name}: {exc}") from exc
        if not text:
            raise ValueError("В документе не найден извлекаемый текст; сканированные PDF требуют OCR")

        mentioned = sorted(set(GID_PATTERN.findall(text)))
        if known_gids is not None:
            mentioned = [gid for gid in mentioned if gid in known_gids]
        record = DocumentRecord(
            document_id=document_id,
            filename=safe_name,
            extension=extension,
            size_bytes=len(content),
            sha256=digest,
            added_at=datetime.now(timezone.utc).isoformat(),
            text_chars=len(text),
            detected_gids=mentioned,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{document_id}{extension}").write_bytes(content)
        (self.root / f"{document_id}.txt").write_text(text, encoding="utf-8")
        metadata_path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        return record, True

    def list_documents(self) -> list[DocumentRecord]:
        if not self.root.exists():
            return []
        records = []
        for path in self.root.glob("*.meta.json"):
            try:
                records.append(DocumentRecord(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(records, key=lambda item: item.added_at, reverse=True)

    def search(self, query: str, limit: int = 10) -> list[dict[str, object]]:
        terms = [term.lower() for term in re.findall(r"[\w\-]{3,}", query, flags=re.UNICODE)]
        if not terms:
            return []
        results = []
        for record in self.list_documents():
            text_path = self.root / f"{record.document_id}.txt"
            if not text_path.exists():
                continue
            text = text_path.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            filename_lowered = record.filename.lower()
            positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
            filename_matches = sum(term in filename_lowered for term in terms)
            if not positions and not filename_matches:
                continue
            position = min(positions) if positions else 0
            start = max(0, position - 180)
            end = min(len(text), position + 420)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            score = sum(lowered.count(term) for term in terms) + 2 * filename_matches
            results.append(
                {
                    "document_id": record.document_id,
                    "filename": record.filename,
                    "score": score,
                    "snippet": snippet,
                    "detected_gids": record.detected_gids,
                }
            )
        ordered = sorted(results, key=lambda item: (-int(item["score"]), str(item["filename"])))
        return ordered[: max(1, min(limit, 25))]
