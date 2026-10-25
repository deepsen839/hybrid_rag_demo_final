import os

from groq import Groq

from rag.config import GROQ_MODEL


def generate_stream(
    query,
    context,
):
    client = Groq(
        api_key=os.environ[
            "GROQ_API_KEY"
        ]
    )

    system_prompt = f"""
You are a grounded PDF RAG assistant.

Use ONLY the supplied document context.

Rules:
1. Never invent facts.
2. Never use external knowledge.
3. If the answer is unsupported, say:
   Context not found in documents.
4. Preserve programming syntax and code.
5. Cite the PDF filename and page for factual claims.
6. If sources conflict, explicitly state the conflict.
7. Do not claim a page supports something unless the
   supplied passage supports it.

DOCUMENT CONTEXT:

{context}
"""

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0.1,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue

        content = (
            chunk.choices[0]
            .delta
            .content
        )

        if content:
            yield content
