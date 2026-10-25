import json
import time

from rag.retrieval import HybridRetriever


def recall_at_k(
    results,
    relevant_pages,
    k,
):
    found = {
        item["payload"]["page"]
        for item in results[:k]
    }

    return int(
        bool(
            found.intersection(
                set(relevant_pages)
            )
        )
    )


def reciprocal_rank(
    results,
    relevant_pages,
):
    relevant_pages = set(
        relevant_pages
    )

    for rank, result in enumerate(
        results,
        1,
    ):
        if (
            result["payload"]["page"]
            in relevant_pages
        ):
            return 1.0 / rank

    return 0.0


def evaluate(
    path="evaluation/test_questions.json"
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    retriever = HybridRetriever()

    rows = []

    for question in questions:
        start = time.perf_counter()

        results, metrics = (
            retriever.search(
                question["question"]
            )
        )

        seconds = (
            time.perf_counter()
            - start
        )

        rows.append({
            "question": question[
                "question"
            ],
            "recall_at_5": recall_at_k(
                results,
                question[
                    "relevant_pages"
                ],
                5,
            ),
            "mrr": reciprocal_rank(
                results,
                question[
                    "relevant_pages"
                ],
            ),
            "retrieval_seconds": round(
                seconds,
                4,
            ),
            **metrics,
        })

    return rows


if __name__ == "__main__":
    rows = evaluate()

    for row in rows:
        print(row)
