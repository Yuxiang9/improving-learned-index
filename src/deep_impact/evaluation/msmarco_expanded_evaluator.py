from __future__ import annotations

"""msmarco_expanded_evaluator.py

Evaluate a DeepImpact-style sparse retriever on the MS MARCO **dev** queries
against an *expanded* passage collection.

The expanded corpus **must** be provided as a local TSV file of the form
```
<doc_id>\t<expanded passage text>\n
```
where `<doc_id>` is the original MS MARCO passage id so that it aligns with
standard qrels.

Queries and qrels are also expected as TSV files:
* queries.tsv  `<query_id>\t<query text>`
* qrels.tsv    `<query_id> 0 <doc_id> <relevance>` (standard MS MARCO format)

Example usage
-------------
python -m deep_impact.evaluation.msmarco_expanded_evaluator \
    --corpus /path/to/expanded_collection.tsv \
    --queries /path/to/msmarco/dev_queries.tsv \
    --qrels /path/to/msmarco/qrels.dev.tsv \
    --model soyuj/deeper-impact \
    --device cuda  

The script prints NDCG, MAP, Recall and Precision at 10/100/1000.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval

from .nano_beir_evaluator import SparseSearch  # reuse existing implementation
from ..models.original import DeepImpact


def _read_corpus_tsv(path: Path) -> Dict[str, str]:
    """Read `<docid>\t<text>` lines into a dict."""
    corpus = {}
    with path.open(encoding="utf-8") as f:
        tsv = csv.reader(f, delimiter="\t")
        for row in tsv:
            if len(row) < 2:
                continue
            pid, text = row[0], row[1]
            if text:
                corpus[pid] = text
    return corpus


def _read_queries_tsv(path: Path) -> Dict[str, str]:
    """Read `<qid>\t<query>` lines into a dict."""
    queries = {}
    with path.open(encoding="utf-8") as f:
        tsv = csv.reader(f, delimiter="\t")
        for row in tsv:
            if len(row) < 2:
                continue
            qid, query = row[0], row[1]
            if query:
                queries[qid] = query
    return queries


def _read_qrels_tsv(path: Path) -> Dict[str, Dict[str, int]]:
    """Read MS MARCO qrels TSV into nested dict required by BEIR."""
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


from torchmetrics.retrieval import RetrievalMRR  # import after torch available


def _compute_mrr(results: Dict[str, Dict[str, float]],
                qrels: Dict[str, Dict[str, int]],
                K_values=(10, 100, 1000)) -> Dict[str, float]:
    """Compute Mean Reciprocal Rank at given cut-offs using torchmetrics."""
    # Flatten results into lists for torchmetrics
    preds = []
    target = []
    indexes = []

    for qid, doc_scores in results.items():
        int_qid = int(qid)  # MS MARCO query ids are numeric strings
        rel_docs = qrels.get(qid, {})
        for docid, score in doc_scores.items():
            preds.append(score)
            target.append(1 if docid in rel_docs and rel_docs[docid] > 0 else 0)
            indexes.append(int_qid)

    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.int)
    index_tensor = torch.tensor(indexes, dtype=torch.long)

    mrr_scores: Dict[str, float] = {}
    for K in K_values:
        metric = RetrievalMRR(top_k=K)
        mrr_value = metric(preds_tensor, target_tensor, index_tensor).item()
        mrr_scores[f"MRR@{K}"] = mrr_value

    return mrr_scores

def evaluate(model, corpus, queries, qrels, *, batch_size: int = 16, device: str = "cpu", verbose: bool = False):
    searcher = SparseSearch(model, batch_size=batch_size, verbose=verbose)
    results = searcher.search(queries, corpus, k=1000)
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [10, 100, 1000])

    # Compute MRR via torchmetrics
    mrr = _compute_mrr(results, qrels, K_values=(10, 100, 1000))

    return ndcg, _map, recall, precision, mrr



def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepImpact on expanded MS MARCO dev corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to expanded corpus TSV file")
    parser.add_argument("--queries", type=Path, required=True, help="Path to dev queries TSV file")
    parser.add_argument("--qrels", type=Path, required=True, help="Path to dev qrels TSV file")
    parser.add_argument("--model", type=str, default="soyuj/deeper-impact", help="HF repo or local path to DeepImpact model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | cuda:0 ...")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.verbose:
        print("Loading data ...")

    corpus = _read_corpus_tsv(args.corpus)
    queries = _read_queries_tsv(args.queries)
    qrels = _read_qrels_tsv(args.qrels)

    if args.verbose:
        print(f"Corpus:  {len(corpus):,} passages")
        print(f"Queries: {len(queries):,}")
        print(f"Qrels:   {sum(len(v) for v in qrels.values()):,} total judgments")
        print("Loading model ...")

    model = DeepImpact.load(args.model)
    model.to(args.device)
    model.eval()

    metrics = evaluate(
        model,
        corpus,
        queries,
        qrels,
        batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main() 