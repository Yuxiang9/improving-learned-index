import gzip
import json
import pickle
from pathlib import Path
from typing import Optional
from typing import Union, Set

from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.defaults import COLLECTION_TYPES
from src.utils.logger import Logger

logger = Logger(__name__)


class Queries:
    """
    Queries dataset.
    :param queries_path: Path to the queries dataset. Each line is a query of (qid, query)
    """

    def __init__(self, queries_path: Union[str, Path], dataset_type: Optional[str] = COLLECTION_TYPES[0]):
        self.dataset_type = dataset_type
        self.queries = self._load_queries(queries_path)

    def _load_queries(self, path):
        queries = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                qid, query = QueryParser.parse(line, self.dataset_type)
                queries[qid] = query
        return queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, qid):
        return self.queries[qid]

    def __iter__(self):
        for qid in self.queries:
            yield qid, self.queries[qid]

    def keys(self):
        return self.queries.keys()


class Collection:
    """
    Collection dataset.
    :param collection_path: Path to the collection dataset. Each line is a passage of (pid, passage)
    """

    def __init__(self, collection_path: Union[str, Path], offset: Optional[int] = None, limit: Optional[int] = None):
        self.collection = self._load_collection(collection_path, offset, limit)

    @staticmethod
    def _load_collection(path, offset: Optional[int] = None, limit: Optional[int] = None):
        if offset is None:
            offset = 0
        if limit is None:
            limit = float('inf')

        collection = {}
        with open(path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx < offset:
                    continue
                if idx >= offset + limit:
                    break
                pid, passage, = line.strip().split('\t')
                # assert int(pid) == idx, "Collection is not sorted by id"
                collection[int(pid)] = passage
        return collection

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, pid):
        return self.collection[pid]

    def __iter__(self):
        for pid in self.collection:
            yield pid, self.collection[pid]

    def batch_iter(self, batch_size: int):
        batch = []
        for pid, passage in self.collection.items():
            batch.append((pid, passage))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch


class MSMarcoTriples(Dataset):
    def __init__(self, triples_path: Union[str, Path], queries_path: Union[str, Path],
                 collection_path: Union[str, Path]):
        """
        Initialize the MS MARCO Triples Dataset object
        :param triples_path: Path to the triples dataset. Each line is a triple of (qid, pos_id, neg_id)
        :param queries_path: Path to the queries dataset. Each line is a query of (qid, query)
        :param collection_path: Path to the collection dataset. Each line is a passage of (pid, passage)
        """
        logger.info(f"Loading triples from {triples_path}")
        self.triples = self._load_triples(triples_path)

        self.queries = Queries(queries_path)
        self.collection = Collection(collection_path)

    @staticmethod
    def _load_triples(path):
        triples = []
        with open(path, encoding='utf-8') as f:
            for line in tqdm(f):
                qid, pos, neg = map(int, line.strip().split("\t"))
                triples.append((qid, pos, neg))
        return triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """
        Get the query, pos and neg items by an index.
        :param idx: The index.
        :return: The query, positive (relevant) doc and negative (non-relevant) doc.
        """
        qid, pos_id, neg_id = self.triples[idx]
        query, positive_doc, negative_doc = self.queries[qid], self.collection[pos_id], self.collection[neg_id]
        return query, positive_doc, negative_doc


class QueryRelevanceDataset:
    def __init__(self, qrels_path: Union[str, Path]):
        """
        Initialize the Qrels Dataset object
        :param qrels_path: Path to the qrels file. Each line: (qid, 0, pid, 1)
        """
        self.qrels = self._load_qrels(qrels_path)

    @staticmethod
    def _load_qrels(path):
        qrels = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, x, pid, y = map(int, line.strip().split('\t'))
                assert x == 0 and y == 1, "Qrels file is not in the expected format"
                qrels.setdefault(qid, set()).add(pid)

        average_positive_per_query = round(sum(len(qrels[qid]) for qid in qrels) / len(qrels), 2)
        logger.info(f"Loaded {len(qrels)} queries with {average_positive_per_query} positive passages/query on average")

        return qrels

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, qid: int) -> Set:
        """
        Get the positive passage ids for a query id.
        :param qid: The query id.
        :return: Set of positive passage ids for the query.
        """
        return self.qrels[qid]

    def keys(self):
        return self.qrels.keys()


class TopKDataset:
    def __init__(self, top_k_path: Union[str, Path]):
        self.queries, self.passages, self.top_k = self._load_topK(top_k_path)

    def _load_topK(self, path):
        queries, passages, top_k = {}, {}, {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, pid, query, passage = line.strip().split('\t')
                qid, pid = int(qid), int(pid)

                assert (qid not in queries) or (queries[qid] == query), "TopK file is not in the expected format"
                queries[qid] = query
                passages[pid] = passage
                top_k.setdefault(qid, []).append(pid)

        assert all(len(top_k[qid]) == len(set(top_k[qid])) for qid in top_k), "TopK file contains duplicates"
        lens = [len(top_k[qid]) for qid in top_k]

        self.min_len = min(lens)
        self.max_len = max(lens)
        self.avg_len = round(sum(lens) / len(top_k), 2)

        logger.info(f"Loaded {len(top_k)} queries.")
        logger.info(f"Top K Passages distribution: min={self.min_len}, max={self.max_len}, avg={self.avg_len}")

        return queries, passages, top_k

    def __len__(self):
        return len(self.top_k)

    def __getitem__(self, qid):
        """
        Get the top k passage ids for a query id.
        :param qid: The query id.
        :return: The top k passage ids for the query.
        """
        return self.top_k[qid]

    def keys(self):
        return self.top_k.keys()


class DistilHardNegatives(MSMarcoTriples):
    @staticmethod
    def _load_triples(path):
        triples = []
        with open(path, encoding='utf-8') as f:
            for line in tqdm(f):
                qid, pos_id, neg_id, pos_score, neg_score = line.strip().split("\t")
                triples.append((int(qid), int(pos_id), int(neg_id), float(pos_score), float(neg_score)))
        return triples

    def __getitem__(self, idx):
        """
        Get the query, pos, neg document and score by an index.
        :param idx: The index.
        :return: The query, positive (relevant) doc and negative (non-relevant) doc, and their scores.
        """
        qid, pos_id, neg_id, pos_score, neg_score = self.triples[idx]
        return self.queries[qid], self.collection[pos_id], self.collection[neg_id], pos_score, neg_score


class DistillationScores:
    def __init__(self, scores_path: Union[str, Path], queries_path: Union[str, Path],
                 collection_path: Union[str, Path], batch_size: int = 55,
                 qrels_path: Optional[Union[str, Path]] = None):
        self.batch_size = batch_size
        self.qrels = qrels_path and QueryRelevanceDataset(qrels_path)
        self.queries = Queries(queries_path)
        self.collection = Collection(collection_path)
        self.dataset = self.construct_dataset(self._load_scores(scores_path))

    def construct_dataset(self, scores):
        # Margin MSE distillation loss
        if self.qrels:
            lookup = []
            for qid in self.qrels.keys():
                positive_docs = [(x, scores[qid].pop(x)) for x in self.qrels[qid]]
                negative_docs = list(scores[qid].items())

                for pos_doc in positive_docs:
                    for i in range(0, len(negative_docs), self.batch_size):
                        if i + self.batch_size <= len(negative_docs):
                            lookup.append((qid, [pos_doc] + negative_docs[i:i + self.batch_size]))
                        else:
                            break
            return lookup
        # KL divergence distillation loss
        else:
            lookup = []
            for qid in scores:
                docs = list(scores[qid].items())
                for i in range(0, len(docs), self.batch_size):
                    if i + self.batch_size <= len(docs):
                        lookup.append((qid, docs[i:i + self.batch_size]))
                    else:
                        break
            return lookup

    @staticmethod
    def _load_scores(path):
        with gzip.open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        qid, pid_score_list = self.dataset[idx]
        return self.queries[qid], [(self.collection[pid], score) for pid, score in pid_score_list]


class RunFile:
    def __init__(self, run_file_path: Union[str, Path]):
        self.run_file_path = run_file_path

    def write(self, qid, pid, rank, score):
        with open(self.run_file_path, 'a', encoding='utf-8') as f:
            f.write(f'{qid}\t{pid}\t{rank}\t{score}\n')

    def writelines(self, qid, scores):
        with open(self.run_file_path, 'a', encoding='utf-8') as f:
            for rank, (pid, score) in enumerate(scores, start=1):
                f.write(f'{qid}\t{pid}\t{rank}\t{score}\n')

    def read(self):
        with open(self.run_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, pid, rank, score = line.strip().split('\t')
                yield int(qid), int(pid), int(rank), float(score)


class TopKRunFile(RunFile):
    def __init__(self, run_file_path: Union[str, Path], k: int = 2000):
        super().__init__(run_file_path)

        top_k = {}
        for qid, pid, rank, _ in self.read():
            top_k.setdefault(qid, []).append((rank, pid))
        for qid in top_k:
            top_k[qid].sort()
            top_k[qid] = [v for _, v in top_k[qid][:k]]
        self.top_k = top_k

    def __len__(self):
        return len(self.top_k)

    def __getitem__(self, qid):
        return self.top_k[qid]

    def __iter__(self):
        for qid in self.top_k:
            yield qid, self.top_k[qid]


class CollectionParser:
    _mapping = {
        'msmarco': 'get_msmarco_item',
        'beir': 'get_beir_item'
    }

    @staticmethod
    def get_msmarco_item(passage):
        return passage.strip().split('\t')

    @staticmethod
    def get_beir_item(passage):
        item = json.loads(passage)
        return item['_id'], item['title'] + ' ' + item['text']

    @staticmethod
    def parse(item, collection_type):
        return getattr(CollectionParser, CollectionParser._mapping[collection_type])(item)


class QueryParser:
    _mapping = {
        'msmarco': 'get_msmarco_item',
        'beir': 'get_beir_item'
    }

    @staticmethod
    def get_msmarco_item(query):
        qid, query = query.strip().split('\t')
        return int(qid), query

    @staticmethod
    def get_beir_item(query):
        item = json.loads(query)
        return item['_id'], item['text']

    @staticmethod
    def parse(item, collection_type):
        return getattr(QueryParser, QueryParser._mapping[collection_type])(item)


class DistillationScoresWithHardNegatives:
    """
    Converts distillation scores to (query, positive, [hard_negatives]) format.
    
    For each query:
    - Positive: highest scored document
    - Hard negatives: K lowest scored documents (default K=8)
    
    This is useful for training with margin loss + contrastive loss + KD.
    """
    def __init__(self, scores_path: Union[str, Path], queries_path: Union[str, Path],
                 collection_path: Union[str, Path], num_hard_negatives: int = 8):
        """
        Initialize the dataset.
        
        :param scores_path: Path to the distillation scores file (.pkl.gz)
        :param queries_path: Path to the queries dataset
        :param collection_path: Path to the collection dataset
        :param num_hard_negatives: Number of hard negatives per query (default: 8)
        """
        logger.info(f"Loading distillation scores from {scores_path}")
        self.queries = Queries(queries_path)
        self.collection = Collection(collection_path)
        self.num_hard_negatives = num_hard_negatives
        self.dataset = self._convert_scores_to_dataset(self._load_scores(scores_path))
        logger.info(f"Created {len(self.dataset)} training examples with 1 positive + {num_hard_negatives} hard negatives each")

    @staticmethod
    def _load_scores(path):
        """Load distillation scores from pkl.gz file."""
        with gzip.open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

    def _convert_scores_to_dataset(self, scores):
        """
        Convert scores to (qid, pos_pid, [neg_pid1, neg_pid2, ...]) format.
        
        For each query:
        - Positive: highest scored document
        - Hard negatives: K lowest scored documents
        """
        dataset = []
        for qid, doc_scores in tqdm(scores.items(), desc="Processing queries"):
            if len(doc_scores) < (1 + self.num_hard_negatives):
                continue  # Need enough docs for 1 pos + K negatives
            
            # Sort documents by score (descending)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Take highest as positive
            pos_pid = sorted_docs[0][0]
            
            # Take K lowest as hard negatives
            neg_pids = [doc[0] for doc in sorted_docs[-self.num_hard_negatives:]]
            
            dataset.append((qid, pos_pid, neg_pids))
        
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get (query, positive_doc, [negative_docs]) by index.
        
        :param idx: The index
        :return: (query, positive_doc, [negative_doc1, negative_doc2, ...])
        """
        qid, pos_id, neg_ids = self.dataset[idx]
        query = self.queries[qid]
        positive_doc = self.collection[pos_id]
        negative_docs = [self.collection[nid] for nid in neg_ids]
        return query, positive_doc, negative_docs


class DistillationScoresToTriples:
    """
    Converts distillation scores (pkl.gz) to positive/negative triples for contrastive learning.
    This is useful for training with InfoNCE loss when you have teacher scores but need pairs.
    
    Strategy: For each query, take the highest scored document as positive and lowest as hard negative.
    """
    def __init__(self, scores_path: Union[str, Path], queries_path: Union[str, Path],
                 collection_path: Union[str, Path]):
        """
        Initialize the dataset by converting scores to triples.
        
        :param scores_path: Path to the distillation scores file (.pkl.gz)
        :param queries_path: Path to the queries dataset
        :param collection_path: Path to the collection dataset
        """
        logger.info(f"Converting distillation scores from {scores_path} to triples")
        self.queries = Queries(queries_path)
        self.collection = Collection(collection_path)
        self.triples = self._convert_scores_to_triples(self._load_scores(scores_path))
        logger.info(f"Created {len(self.triples)} training triples from distillation scores")

    @staticmethod
    def _load_scores(path):
        """Load distillation scores from pkl.gz file."""
        with gzip.open(path, 'rb') as f:
            scores = pickle.load(f)
        return scores

    def _convert_scores_to_triples(self, scores):
        """
        Convert scores to (qid, pos_pid, neg_pid) triples.
        
        For each query:
        - Positive: highest scored document
        - Negative: lowest scored document (hard negative)
        """
        triples = []
        for qid, doc_scores in tqdm(scores.items(), desc="Converting to triples"):
            if len(doc_scores) < 2:
                continue  # Need at least 2 docs (pos and neg)
            
            # Sort documents by score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Take highest as positive, lowest as negative
            pos_pid = sorted_docs[0][0]
            neg_pid = sorted_docs[-1][0]
            
            triples.append((qid, pos_pid, neg_pid))
        
        return triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        """
        Get the query, positive doc, and negative doc by index.
        
        :param idx: The index
        :return: (query, positive_doc, negative_doc) tuple
        """
        qid, pos_id, neg_id = self.triples[idx]
        query = self.queries[qid]
        positive_doc = self.collection[pos_id]
        negative_doc = self.collection[neg_id]
        return query, positive_doc, negative_doc
