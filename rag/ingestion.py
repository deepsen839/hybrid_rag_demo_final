import hashlib
import io
import os
import re
import uuid
from collections import Counter

import fitz
import pytesseract
from PIL import Image
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer
from qdrant_client.models import PointStruct

from rag.config import (
    EMBEDDING_MODEL,
    CHUNK_TOKENS,
    CHUNK_OVERLAP,
)
from rag.chunking import TokenChunker
from rag.embeddings import Embeddings


class PDFIngestor:
    def __init__(self):
        self.embeddings = Embeddings()

        self.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL
        )

        self.chunker = TokenChunker(
            self.tokenizer,
            CHUNK_TOKENS,
            CHUNK_OVERLAP,
        )

        self.detector = (
            LanguageDetectorBuilder
            .from_languages(
                Language.ENGLISH,
                Language.GERMAN,
                Language.FRENCH,
                Language.SPANISH,
                Language.HINDI,
            )
            .build()
        )

    def document_id(self, data):
        return hashlib.sha256(data).hexdigest()

    def detect_language(self, text):
        try:
            result = self.detector.detect_language_of(
                text[:4000]
            )
            if result:
                return result.iso_code_639_1.name.lower()
        except Exception:
            pass

        return "unknown"

    def native_blocks(self, page):
        blocks = []

        for block in page.get_text("blocks"):
            if len(block) < 7:
                continue

            x0, y0, x1, y1, text, block_no, block_type = (
                block[:7]
            )

            text = text.strip()

            if not text:
                continue

            blocks.append({
                "text": text,
                "bbox": [
                    round(x0, 2),
                    round(y0, 2),
                    round(x1, 2),
                    round(y1, 2),
                ],
                "block_no": int(block_no),
                "block_type": int(block_type),
            })

        return blocks

    def ocr_page(self, page):
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False,
        )

        image = Image.open(
            io.BytesIO(
                pix.tobytes("png")
            )
        ).convert("RGB")

        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--psm 3",
        )

        words = []

        for i, word in enumerate(data["text"]):
            word = word.strip()

            if not word:
                continue

            try:
                confidence = float(
                    data["conf"][i]
                )
            except Exception:
                confidence = -1

            if confidence < 0:
                continue

            x = data["left"][i] / 2
            y = data["top"][i] / 2
            w = data["width"][i] / 2
            h = data["height"][i] / 2

            words.append({
                "text": word,
                "bbox": [
                    round(x, 2),
                    round(y, 2),
                    round(x + w, 2),
                    round(y + h, 2),
                ],
            })

        text = " ".join(
            w["text"] for w in words
        )

        bbox = None

        if words:
            bbox = [
                min(w["bbox"][0] for w in words),
                min(w["bbox"][1] for w in words),
                max(w["bbox"][2] for w in words),
                max(w["bbox"][3] for w in words),
            ]

        return text.strip(), bbox

    def remove_repeated_headers_footers(
        self,
        page_blocks,
        total_pages,
    ):
        if total_pages < 5:
            return page_blocks

        top_counter = Counter()
        bottom_counter = Counter()

        for blocks in page_blocks:
            if not blocks:
                continue

            page_height = max(
                (
                    b["bbox"][3]
                    for b in blocks
                ),
                default=1,
            )

            for block in blocks:
                normalized = re.sub(
                    r"\s+",
                    " ",
                    block["text"].lower(),
                ).strip()

                if (
                    block["bbox"][1]
                    <= page_height * 0.12
                ):
                    top_counter[
                        normalized
                    ] += 1

                if (
                    block["bbox"][3]
                    >= page_height * 0.88
                ):
                    bottom_counter[
                        normalized
                    ] += 1

        threshold = max(
            3,
            int(total_pages * 0.60),
        )

        repeated_top = {
            text
            for text, count in top_counter.items()
            if count >= threshold
            and len(text) <= 200
        }

        repeated_bottom = {
            text
            for text, count in bottom_counter.items()
            if count >= threshold
            and len(text) <= 200
        }

        cleaned = []

        for blocks in page_blocks:
            current = []

            for block in blocks:
                normalized = re.sub(
                    r"\s+",
                    " ",
                    block["text"].lower(),
                ).strip()

                if (
                    normalized in repeated_top
                    or normalized in repeated_bottom
                ):
                    continue

                current.append(block)

            cleaned.append(current)

        return cleaned

    def index_pdf(
        self,
        uploaded_file,
        retriever,
    ):
        data = uploaded_file.getvalue()
        document_id = self.document_id(data)

        if retriever.document_exists(
            document_id
        ):
            return 0, 0

        filename = uploaded_file.name

        doc = fitz.open(
            stream=data,
            filetype="pdf",
        )

        total_pages = len(doc)

        page_blocks = []
        ocr_pages = set()

        # ---------------------------------------------
        # Extraction
        # ---------------------------------------------
        for page_number in range(
            total_pages
        ):
            page = doc[page_number]

            blocks = self.native_blocks(
                page
            )

            native_text = "\n".join(
                b["text"]
                for b in blocks
            ).strip()

            # OCR fallback
            if len(
                re.sub(
                    r"\W+",
                    "",
                    native_text,
                )
            ) < 30:

                ocr_text, ocr_bbox = (
                    self.ocr_page(page)
                )

                if ocr_text:
                    ocr_pages.add(
                        page_number
                    )

                    blocks = [{
                        "text": ocr_text,
                        "bbox": (
                            ocr_bbox
                            or [0, 0, 0, 0]
                        ),
                        "block_no": 0,
                        "block_type": 1,
                    }]

            page_blocks.append(
                blocks
            )

        # ---------------------------------------------
        # Header/footer cleaning
        # ---------------------------------------------
        page_blocks = (
            self.remove_repeated_headers_footers(
                page_blocks,
                total_pages,
            )
        )

        points = []
        bm25_records = []

        # ---------------------------------------------
        # Chunk + embed
        # ---------------------------------------------
        for page_number, blocks in enumerate(
            page_blocks
        ):
            if not blocks:
                continue

            page_text = "\n".join(
                b["text"]
                for b in blocks
            ).strip()

            if not page_text:
                continue

            language = (
                self.detect_language(
                    page_text
                )
            )

            content_type = (
                "ocr"
                if page_number in ocr_pages
                else "native"
            )

            chunks = self.chunker.chunk(
                page_text
            )

            embeddings = (
                self.embeddings.encode(
                    chunks
                )
            )

            for chunk_index, (
                chunk,
                embedding,
            ) in enumerate(
                zip(
                    chunks,
                    embeddings,
                )
            ):
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        (
                            f"{document_id}:"
                            f"{page_number + 1}:"
                            f"{chunk_index}"
                        ),
                    )
                )

                bbox = [
                    min(
                        b["bbox"][0]
                        for b in blocks
                    ),
                    min(
                        b["bbox"][1]
                        for b in blocks
                    ),
                    max(
                        b["bbox"][2]
                        for b in blocks
                    ),
                    max(
                        b["bbox"][3]
                        for b in blocks
                    ),
                ]

                payload = {
                    "chunk_id": point_id,
                    "document_id": document_id,
                    "source": filename,
                    "page": page_number + 1,
                    "chunk_index": chunk_index,
                    "language": language,
                    "content_type": content_type,
                    "bbox": bbox,
                    "text": chunk,
                }

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                )

                bm25_records.append(
                    payload
                )

        doc.close()

        retriever.upsert_points(
            points
        )

        retriever.add_bm25_records(
            bm25_records
        )

        return (
            total_pages,
            len(points),
        )
