from pathlib import Path
from collections import Counter, defaultdict
from typing import Union, Optional

from tqdm.auto import tqdm

from src.utils.datasets import QueryRelevanceDataset, RunFile


def compute_first_hit_distribution(
        run_file_path: Union[str, Path],
        qrels_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        consider_unjudged: bool = False,
):
    """Compute the distribution of the first positive document rank for each query.

    Parameters
    ----------
    run_file_path: str or Path
        Path to the run file produced by ``Ranker``. Each line is ``qid pid rank score``.
    qrels_path: str or Path
        Path to the qrels file. Used to identify positive passages for each query.
    output_path: str or Path, optional
        If provided, write the distribution as ``rank,count`` pairs to this file. Otherwise, prints to stdout.
    consider_unjudged: bool, default False
        If True, queries that never retrieve a judged positive document are counted under rank ``None``.
    """

    run_reader = RunFile(run_file_path)
    qrels = QueryRelevanceDataset(qrels_path)

    # map qid -> first positive doc rank
    first_hit_rank = {}

    # For fast lookup
    positive_lookup = {qid: positives for qid, positives in qrels.qrels.items()}

    for qid, pid, rank, _ in tqdm(run_reader.read(), desc="Scanning run file"):
        if qid in first_hit_rank:  # already found first positive
            continue
        if pid in positive_lookup.get(qid, set()):
            first_hit_rank[qid] = rank

    if consider_unjudged:
        for qid in qrels.keys():
            if qid not in first_hit_rank:
                first_hit_rank[qid] = None

    # Build distribution
    distribution = Counter(first_hit_rank.values())
    total_queries = sum(distribution.values())

    # Output
    def _write_line(out, rank, count):
        # print(type(count), type(total_queries))
        percent = int(count) / int(total_queries) * 100 if total_queries > 0 else 0
        out.write(f"{rank}\t{count}\t{percent:.2f}%\n")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write header without attempting percentage computation
            f.write("#rank\tcount\tpercentage\n")
            for rank, count in sorted(distribution.items(), key=lambda x: (x[0] is None, x[0])):
                _write_line(f, rank, count)
    else:
        print("Rank\tCount\tPercentage")
        for rank, count in sorted(distribution.items(), key=lambda x: (x[0] is None, x[0])):
            percent = count / total_queries * 100 if total_queries > 0 else 0
            print(f"{rank}\t{count}\t{percent:.2f}%")

    # # Create plots
    # import matplotlib.pyplot as plt
    # import numpy as np

    # # Filter out None values and sort by rank
    # valid_dist = [(r,c) for r,c in distribution.items() if r is not None]
    # ranks, counts = zip(*sorted(valid_dist))
    # percentages = [c/total_queries * 100 for c in counts]

    # # Create figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    # # Plot rank vs count
    # ax1.bar(ranks, counts)
    # ax1.set_xlabel('Rank')
    # ax1.set_ylabel('Count')
    # ax1.set_title('Distribution of First Positive Hit by Rank')
    # ax1.grid(True, alpha=0.3)

    # # Plot rank vs percentage
    # ax2.bar(ranks, percentages)
    # ax2.set_xlabel('Rank')
    # ax2.set_ylabel('Percentage')
    # ax2.set_title('Distribution of First Positive Hit by Percentage')
    # ax2.grid(True, alpha=0.3)

    # plot_path = "/scratch/yx3044/Projects/improving-learned-index/first_hit_distribution.png"
    # plt.savefig(plot_path)
    # plt.close()



if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="Compute distribution of first retrieved positive passage per query.")
    # parser.add_argument("--run_file_path", help="Path to run file produced by Ranker")
    # parser.add_argument("--qrels_path", help="Path to qrels file")
    # parser.add_argument("--output_path", "-o", help="Optional output file path to write distribution")
    # parser.add_argument("--consider-unjudged", action="store_true",
    #                     help="Include queries with no retrieved positive passage as rank None")

    # args = parser.parse_args()

    run_file_path = "/scratch/yx3044/Projects/improving-learned-index/llama3_expanded_sep3_ranked"
    qrels_path = "/scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/qrels.dev.small.tsv"
    output_path = "/scratch/yx3044/Projects/improving-learned-index/first_hit_distribution.tsv"
    consider_unjudged = True

    compute_first_hit_distribution(run_file_path, qrels_path, output_path, consider_unjudged)
