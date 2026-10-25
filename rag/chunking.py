import re


class TokenChunker:
    def __init__(
        self,
        tokenizer,
        chunk_tokens=450,
        overlap=75,
    ):
        if overlap >= chunk_tokens:
            raise ValueError(
                "overlap must be smaller than chunk_tokens"
            )

        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap

    def clean(self, text):
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk(self, text):
        text = self.clean(text)

        if not text:
            return []

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        chunks = []
        start = 0

        while start < len(token_ids):
            end = min(
                start + self.chunk_tokens,
                len(token_ids),
            )

            ids = token_ids[start:end]

            chunk = self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(token_ids):
                break

            start = end - self.overlap

        return chunks
