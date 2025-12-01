from __future__ import annotations

import logging
from typing import Literal
from collections import defaultdict
import heapq

from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset
import torch
from tqdm import tqdm
import pandas as pd

# from src.deep_impact.evaluation.nano_beir_evaluator import *



class Dataset:
    def __init__(self, queries, corpus, relevant_docs, name):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.name = name


class SparseSearch:
    def __init__(self, model, batch_size, verbose=False):
        self.model = model
        self.batch_size = batch_size
        self.inverted_index = defaultdict(list)  # term_id -> [(doc_id, score), ...]
        self.corpus_ids = []
        self.verbose = verbose
        
    def _build_inverted_index(self, corpus):
        """Build inverted index from corpus embeddings"""
        if self.verbose:
            print(f"Building inverted index for {len(corpus)} documents...")
        
        corpus_ids = list(corpus.keys())
        self.corpus_ids = corpus_ids
        
        # Process corpus in batches
        iterator = tqdm(range(0, len(corpus), self.batch_size), desc="Building inverted index") if self.verbose else range(0, len(corpus), self.batch_size)
        for i in iterator:
            batch_texts = list(corpus.values())[i:i+self.batch_size]
            batch_ids = corpus_ids[i:i+self.batch_size]
            
            with torch.no_grad():
                embeddings = self.model.get_impact_scores_batch(batch_texts)
            
            # Process each document's embedding
            for doc_id, embedding in zip(batch_ids, embeddings):
                for term_id, score in embedding:
                    if score > 0:  # Only store non-zero scores
                        self.inverted_index[term_id].append((doc_id, score))        
        if self.verbose:
            print(f"Built inverted index with {len(self.inverted_index)} terms")
        
    def search(self, queries, corpus, k):
        # Build inverted index if not already built
        if not self.inverted_index:
            self._build_inverted_index(corpus)
        
        results = {}
        if self.verbose:
            print(f"Searching for {len(queries)} queries...")
        
        iterator = tqdm(queries.items(), desc="Searching queries") if self.verbose else queries.items()
        for query_id, query in iterator:
            query_terms = self.model.process_query(query)
            
            doc_scores = defaultdict(float)            
            # Score documents using inverted index
            for query_term in query_terms:
                if query_term in self.inverted_index:
                    for doc_id, doc_score in self.inverted_index[query_term]:
                        doc_scores[doc_id] += doc_score  # Impact score multiplication
            
            # Get top-k documents for this query
            if len(doc_scores) == 0:
                results[query_id] = {}
            else:
                # Use heapq to efficiently get top-k
                if k >= len(doc_scores):
                    top_k_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                else:
                    top_k_docs = heapq.nlargest(k, doc_scores.items(), key=lambda x: x[1])
                
                results[query_id] = {doc_id: float(score) for doc_id, score in top_k_docs}
        
        if self.verbose:
            print(f"Retrieved top-{k} documents for {len(queries)} queries")
        return results

class BaseEvaluator:
    def __init__(self, batch_size=16, verbose=False):
        self.verbose = verbose
        self.batch_size = batch_size
        
    def _load_dataset(self, dataset_name: DatasetNameType) -> Dataset:
        pass
    
    def evaluate_dataset(self, model, dataset_name):
        pass
    
    def evaluate_all(self, model):
        pass

class MSMARCOEvaluator(BaseEvaluator):
    def __init__(self, batch_size=16, verbose=False):
        super().__init__(batch_size, verbose)
        
    def _load_dataset(
        self, dataset_name: DatasetNameType,
    ) -> Dataset:

        if self.verbose:
            print(f"Loading dataset {dataset_name}...")

        corpus_path = "/scratch/yx3044/Projects/improving-learned-index/expansions/lol_expanded.tsv"
        corpus = pd.read_csv(corpus_path, sep="\t")
        queries_path = "/scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/queries.dev.tsv"
        queries = pd.read_csv(queries_path, sep="\t")
        qrels_path = "/scratch/yx3044/Projects/improving-learned-index/required_files/collections/msmarco-passage/qrels.dev.small.tsv"
        qrels = pd.read_csv(qrels_path, sep="\t")
        corpus_dict = {
            sample["_id"]: sample["text"]
            for sample in corpus
            if len(sample["text"]) > 0
        }
        queries_dict = {
            sample["_id"]: sample["text"]
            for sample in queries
            if len(sample["text"]) > 0
        }
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = {}
            qrels_dict[sample["query-id"]][sample["corpus-id"]] = 1

        human_readable_name = MAPPING_DATASET_NAME_TO_HUMAN_READABLE[dataset_name]
        return Dataset(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
        )
        
    def evaluate_all(self, model):
        metrics = self.evaluate_dataset(model, "msmarco")
        print(f"Metrics for msmarco: {metrics}")
        return metrics
            
    def evaluate_dataset(self, model, dataset_name):
        dataset = self._load_dataset(dataset_name)
        searcher = SparseSearch(model, batch_size=self.batch_size, verbose=self.verbose)
        results = searcher.search(dataset.queries, dataset.corpus, k=1000)
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(dataset.relevant_docs, results, [10, 100, 1000])
        return metrics
            
    
    
if __name__ == "__main__":
    from src.deep_impact.models.original import DeepImpact
    model = DeepImpact.load('soyuj/deeper-impact')
    model.to('cuda')
    model.eval()
    evaluator = MSMARCOEvaluator(verbose=True, batch_size=16)
    metrics = evaluator.evaluate_all(model)
    print(metrics)
    
            