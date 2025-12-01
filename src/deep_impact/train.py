from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.deep_impact.models import DeepImpact, DeepPairwiseImpact, DeepImpactCrossEncoder, HybridDeepImpact
from src.deep_impact.training import Trainer, PairwiseTrainer, CrossEncoderTrainer, DistilTrainer, \
    InBatchNegativesTrainer, HybridTrainer
from src.deep_impact.training.distil_trainer import DistilMarginMSE, DistilKLLoss
from src.deep_impact.models.hybrid_deepimpact import EncodingLike
from src.utils.datasets import MSMarcoTriples, DistillationScores, DistillationScoresToTriples
from src.deep_impact.evaluation.nano_beir_evaluator import NanoBEIREvaluator
from src.deep_impact.training.dense_trainer import DenseTrainer, dense_joint_collate_fn, DenseBiEncoder



def collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks = [], []
    for query, positive_document, negative_document in batch:
        encoded_token, mask = model_cls.process_query_and_document(query, positive_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)

        encoded_token, mask = model_cls.process_query_and_document(query, negative_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
    }


def cross_encoder_collate_fn(batch):
    encoded_list = []
    for query, positive_document, negative_document in batch:
        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(positive_document, query)
        encoded_list.append(encoded_token)

        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(negative_document, query)
        encoded_list.append(encoded_token)

    return {'encoded_list': encoded_list}


def distil_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks, scores = [], [], []
    for query, pid_score_list in batch:
        for passage, score in pid_score_list:
            encoded_token, mask = model_cls.process_query_and_document(query, passage, max_length=max_length)
            encoded_list.append(encoded_token)
            masks.append(mask)
            scores.append(score)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
        'scores': torch.tensor(scores, dtype=torch.float),
    }


def in_batch_negatives_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    queries, positive_documents, negative_documents = zip(*batch)
    queries_terms = [model_cls.process_query(query) for query in queries]
    negatives = [model_cls.process_document(document) for document in negative_documents]

    encoded_list, masks = [], []
    for i, (query_terms, positive_document) in enumerate(zip(queries_terms, positive_documents)):
        encoded_token, term_to_token_index = model_cls.process_document(positive_document)

        encoded_list.append(encoded_token)
        masks.append(model_cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length))

        encoded_list.append(negatives[i][0])
        for _, term_to_token_index in negatives:
            masks.append(model_cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length))

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0),
    }


def hybrid_collate_fn(batch, model_cls=DeepImpact, max_length=None, num_hard_negatives=8, top_k_hard=4, qrels_path=None):
    """
    Collate function for HybridDeepImpact training with document sampling.
    
    For pure KL divergence loss, we don't need to distinguish between positive and negative.
    We simply sample documents based on their teacher scores (keeping top-K and random samples).
    
    Args:
        batch: List of (query, pid_score_list) tuples
        model_cls: Model class (should be HybridDeepImpact)
        max_length: Maximum sequence length
        num_hard_negatives: Total number of documents to sample per query
        top_k_hard: Number of top-scored documents to always include
        qrels_path: Not used (kept for compatibility)
        
    Returns:
        Dictionary with:
        - encoded_list: List of encoded documents with expansion tokens
        - masks: Sparse masks for overlapping terms [batch*num_docs, seq_len, 1]
        - doc_dense_masks: Dense masks for expansion tokens [batch*num_docs, seq_len, 1]
        - query_dense_masks: Dense masks for query tokens [batch, seq_len, 1]
        - scores: Teacher scores from cross-encoder [batch*num_docs]
        - num_docs_per_query: Number of documents per query (for reshaping)
    """
    import random
    
    encoded_list = []
    sparse_masks = []
    doc_dense_masks = []
    scores = []
    queries = []
    
    # ===============================
    # 1. DOCUMENT SAMPLING (for KL divergence - no pos/neg distinction needed)
    # ===============================
    for query, pid_score_list in batch:
        queries.append(query)
        
        # Sample documents: top-K highest scored + random from remaining
        # Note: pid_score_list is already sorted by score (highest first)
        if len(pid_score_list) <= num_hard_negatives:
            selected = pid_score_list
        else:
            # Top-K highest scored documents (includes positive if it's high-scored)
            top_docs = pid_score_list[:top_k_hard]
            # Random sample from remaining
            remaining = pid_score_list[top_k_hard:]
            n_rand = num_hard_negatives - top_k_hard
            rand_sample = random.sample(remaining, min(n_rand, len(remaining)))
            selected = top_docs + rand_sample
        
        # Extract docs and scores
        docs = [doc for doc, _ in selected]
        doc_scores = [score for _, score in selected]
        
        # Process each document with expansion tokens
        for doc in docs:
            encoded_token, sparse_mask, _, doc_dense_mask = model_cls.process_query_and_document(
                query, doc, max_length=max_length
            )
            encoded_list.append(encoded_token)
            sparse_masks.append(sparse_mask)
            doc_dense_masks.append(doc_dense_mask)
        
        scores.extend(doc_scores)
    
    num_docs_per_query = len(docs)  # Same for all queries in batch
    
    # ===============================
    # 2. PROCESS QUERIES FOR DENSE MASKS AND EMBEDDINGS
    # ===============================
    # Encode queries to get IDs and masks (needed for computing query embeddings)
    tokenizer = model_cls.tokenizer
    query_encodings = tokenizer.batch_encode_plus(
        queries,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Query dense mask: all non-padding tokens
    query_dense_masks = (query_encodings["input_ids"] != tokenizer.pad_token_id).float().unsqueeze(-1)
    
    # Convert query encodings to the same format as encoded_list for consistency
    # We need to create encoding-like objects from the tensors
    # (tokenizers.Encoding constructor changed, can't use keyword args directly)
    query_encoded_list = []
    for i in range(len(queries)):
        query_enc = EncodingLike(
            ids=query_encodings["input_ids"][i].tolist(),
            attention_mask=query_encodings["attention_mask"][i].tolist(),
            type_ids=query_encodings["token_type_ids"][i].tolist() if "token_type_ids" in query_encodings else [0] * max_length,
        )
        query_encoded_list.append(query_enc)
    
    return {
        'encoded_list': encoded_list,
        'query_encoded_list': query_encoded_list,
        'masks': torch.stack(sparse_masks, dim=0).unsqueeze(-1),
        'doc_dense_masks': torch.stack(doc_dense_masks, dim=0).unsqueeze(-1),
        'query_dense_masks': query_dense_masks,
        'scores': torch.tensor(scores, dtype=torch.float),
        'num_docs_per_query': num_docs_per_query,
    }

def run(
        dataset_path: Union[str, Path],
        queries_path: Union[str, Path],
        collection_path: Union[str, Path],
        checkpoint_dir: Union[str, Path],
        max_length: int,
        seed: int,
        batch_size: int,
        lr: float,
        save_every: int,
        save_best: bool,
        gradient_accumulation_steps: int,
        pairwise: bool = False,
        cross_encoder: bool = False,
        distil_mse: bool = False,
        distil_kl: bool = False,
        dense: bool = False,
        in_batch_negatives: bool = False,
        hybrid: bool = False,
        expansion_weight: float = 0.3,
        regular_weight: float = 0.7,
        start_with: Union[str, Path] = None,
        qrels_path: Union[str, Path] = None,
        eval_every: int = 500,
        # HybridDeepImpact specific parameters
        dense_dim: int = 128,
        num_expansion_tokens: int = 64,
        sparse_loss_weight: float = 1.0,
        dense_loss_weight: float = 1.0,
        combined_loss_weight: float = 1.0,
        loss_temperature: float = 1.0,
        fixed_weights: bool = False,
        # HybridDeepImpact specific parameters
        lambda_di: float = 1.0,
        lambda_li: float = 1.0,
        lambda_kd: float = 1.0,
        margin: float = 0.2,
        kd_temperature: float = 1.0,
        num_hard_negatives: int = 8,
        top_k_hard: int = 4,
):
    # DeepImpact
    model_cls = DeepImpact
    trainer_cls = Trainer
    collate_function = partial(collate_fn, model_cls=DeepImpact, max_length=max_length)
    dataset_cls = MSMarcoTriples

    # Pairwise
    if pairwise:
        model_cls = DeepPairwiseImpact
        trainer_cls = PairwiseTrainer
        collate_function = partial(collate_fn, model_cls=DeepPairwiseImpact, max_length=max_length)

    # CrossEncoder
    elif cross_encoder:
        model_cls = DeepImpactCrossEncoder
        trainer_cls = CrossEncoderTrainer
        collate_function = cross_encoder_collate_fn

    elif dense:
        model_cls = DeepImpact
        trainer_cls = DenseTrainer
        trainer_cls.loss = DistilKLLoss()
        collate_function = partial(dense_joint_collate_fn, max_length_sparse=max_length)
        dataset_cls = DistillationScores

    # Hybrid DeepImpact and late interaction with Knowledge Distillation (DeeperImpact-style)
    if hybrid:
        model_cls = HybridDeepImpact
        trainer_cls = HybridTrainer
        collate_function = partial(
            hybrid_collate_fn, 
            model_cls=HybridDeepImpact, 
            max_length=max_length,
            num_hard_negatives=num_hard_negatives,
            top_k_hard=top_k_hard,
        )
        dataset_cls = DistillationScores
    
        
    # Use distillation loss
    elif distil_mse:
        trainer_cls = DistilTrainer
        trainer_cls.loss = DistilMarginMSE()
        collate_function = partial(distil_collate_fn, max_length=max_length)
        dataset_cls = partial(DistillationScores, qrels_path=qrels_path)
    elif distil_kl:
        trainer_cls = DistilTrainer
        trainer_cls.loss = DistilKLLoss()
        collate_function = partial(distil_collate_fn, max_length=max_length)
        dataset_cls = DistillationScores

    if in_batch_negatives:
        trainer_cls = InBatchNegativesTrainer
        collate_function = partial(in_batch_negatives_collate_fn, max_length=max_length)

    trainer_cls.ddp_setup()

    
    # Determine whether to learn weights (default True unless --fixed_weights is specified)
    learn_weights = not fixed_weights if 'fixed_weights' in locals() else True
    
    if start_with:
        if model_cls == HybridDeepImpact:
            model = model_cls.load(start_with, dense_dim=dense_dim, 
                                 num_expansion_tokens=num_expansion_tokens,
                                 learn_weights=learn_weights)
        else:
            model = model_cls.load(start_with)
    else:
        # For HybridDeepImpact, we need to pass configuration parameters
        if model_cls == HybridDeepImpact:
            from transformers import AutoConfig

            print(f"Setting {num_expansion_tokens} expansion tokens...")
            model_cls.configure_expansion_token(num_expansion_tokens)

            # Re-enable padding/truncation AFTER adding tokens
            # model_cls.tokenizer.enable_truncation(max_length=max_length)
            # model_cls.tokenizer.enable_padding(length=max_length)
            model_cls.tokenizer.model_max_length = max_length

            config = AutoConfig.from_pretrained("Luyu/co-condenser-marco")
            model = model_cls(
                config,
                dense_dim=dense_dim,
                num_expansion_tokens=num_expansion_tokens,
                learn_weights=learn_weights,
            )

            # tokenizer.vocab_size ignores newly added tokens (e.g., [EXP]); len() reflects real size
            vocab_size = len(model_cls.tokenizer)
            if vocab_size != model.bert.embeddings.word_embeddings.num_embeddings:
                print(f"Resizing token embeddings {model.bert.embeddings.word_embeddings.num_embeddings} -> {vocab_size}")
                model.resize_token_embeddings(vocab_size)

            # Load base DeepImpact for weight initialization
            base_model = DeepImpact.load()

            # Copy BERT weights safely
            base_state = base_model.bert.state_dict()
            model_state = model.bert.state_dict()
            for key, val in base_state.items():
                if key.startswith("embeddings.word_embeddings.weight"):
                    min_vocab = min(val.size(0), model_state[key].size(0))
                    model_state[key][:min_vocab] = val[:min_vocab]
                elif key in model_state and val.size() == model_state[key].size():
                    model_state[key] = val

            model.bert.load_state_dict(model_state)

            # Copy sparse head
            model.impact_score_encoder.load_state_dict(
                base_model.impact_score_encoder.state_dict()
            )

            print("✓ HybridDeepImpact initialized correctly!")
            
            # Print configuration
            if learn_weights:
                print(f"✓ HybridDeepImpact initialized with LEARNABLE weights (will adapt during training)")
            else:
                print(f"✓ HybridDeepImpact initialized with FIXED weights (0.5/0.5, will NOT change during training)")
        # For HybridDeepImpact, we need to pass the weights during initialization
        elif model_cls == HybridDeepImpact:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained('Luyu/co-condenser-marco')
            model = model_cls(config, expansion_weight=expansion_weight, regular_weight=regular_weight)
            # Load pretrained weights from base model
            base_model = DeepImpact.load()
            # Copy BERT weights
            model.bert.load_state_dict(base_model.bert.state_dict())
            # Initialize regular impact encoder from base model
            model.regular_impact_score_encoder.load_state_dict(base_model.impact_score_encoder.state_dict())
        else:
            model = model_cls.load()
    
    # model_cls.tokenizer.enable_truncation(max_length=max_length, strategy='longest_first')
    # model_cls.tokenizer.enable_padding(length=max_length)
    model_cls.tokenizer.model_max_length = max_length

    dataset = dataset_cls(dataset_path, queries_path, collection_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_function,
        sampler=DistributedSampler(dataset),
        drop_last=True,
        num_workers=0,
    )



    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    evaluator = NanoBEIREvaluator(batch_size=64, verbose=False)
    
    # --------------------------------------------------------------
    # Instantiate the appropriate trainer.
    # DenseTrainer expects *two* separate models (sparse & dense) so
    # we branch here accordingly; for all other trainers we keep the
    # original call signature.
    # --------------------------------------------------------------

    if trainer_cls is DenseTrainer:
        dense_model = DenseBiEncoder()
        trainer = trainer_cls(
            sparse_model=model,
            dense_model=dense_model,
            optimizer=optimizer,
            train_data=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
        )
    elif trainer_cls is HybridTrainer:
        trainer = trainer_cls(
            model=model,
            optimizer=optimizer,
            train_data=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
            lambda_di=lambda_di,
            lambda_li=lambda_li,
            lambda_kd=lambda_kd,
            margin=margin,
            temperature=kd_temperature,
        )
    else:
        trainer = trainer_cls(
            model=model,
            optimizer=optimizer,
            train_data=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
        )
    trainer.train()
    trainer_cls.ddp_cleanup()


if __name__ == "__main__":
    import argparse
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser("Distributed Training of DeepImpact on MS MARCO triples dataset")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the training dataset")
    parser.add_argument("--queries_path", type=Path, required=True, help="Path to the queries dataset")
    parser.add_argument("--collection_path", type=Path, required=True, help="Path to the collection dataset")
    parser.add_argument("--checkpoint_dir", type=Path, required=True, help="Directory to store and load checkpoints")
    parser.add_argument("--max_length", type=int, default=300, help="Max Number of tokens in document")
    parser.add_argument("--seed", type=int, default=42, help="Fix seed")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=20000, help="Save checkpoint every n steps")
    parser.add_argument("--save_best", action="store_true", help="Save the best model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--pairwise", action="store_true", help="Use pairwise training")
    parser.add_argument("--cross_encoder", action="store_true", help="Use cross encoder model")
    parser.add_argument("--distil_mse", action="store_true", help="Use distillation loss with Mean Squared Error")
    parser.add_argument("--distil_kl", action="store_true", help="Use distillation loss with KL divergence loss")
    parser.add_argument("--dense", action="store_true", help="Use dense training")
    parser.add_argument("--in_batch_negatives", action="store_true", help="Use in-batch negatives")
    parser.add_argument("--hybrid", action="store_true", help="Use HybridDeepImpact with KD")
    parser.add_argument("--fixed_weights", action="store_true", help="Use fixed combination weights (0.5/0.5) instead of learning them")
    parser.add_argument("--expansion_weight", type=float, default=0.3, help="Initial weight for expansion token scores (w2)")
    parser.add_argument("--regular_weight", type=float, default=0.7, help="Initial weight for regular term scores (w1)")
    parser.add_argument("--start_with", type=Path, default=None, help="Start training with this checkpoint")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every n steps")
    
    # HybridDeepImpact specific parameters
    parser.add_argument("--dense_dim", type=int, default=128, help="Dimensionality of dense embeddings for HybridDeepImpact")
    parser.add_argument("--num_expansion_tokens", type=int, default=64, help="Number of expansion tokens for HybridDeepImpact")
    parser.add_argument("--sparse_loss_weight", type=float, default=1.0, help="Weight for sparse loss in HybridDeepImpact")
    parser.add_argument("--dense_loss_weight", type=float, default=1.0, help="Weight for dense loss in HybridDeepImpact")
    parser.add_argument("--combined_loss_weight", type=float, default=1.0, help="Weight for combined loss in HybridDeepImpact")
    parser.add_argument("--loss_temperature", type=float, default=1.0, help="Temperature for InfoNCE loss in HybridDeepImpact")
    
    # HybridDeepImpact specific parameters (KL divergence distillation)
    parser.add_argument("--lambda_di", type=float, default=1.0, help="Weight for sparse component KD loss (teacher → sparse)")
    parser.add_argument("--lambda_li", type=float, default=1.0, help="Weight for dense component KD loss (teacher → dense)")
    parser.add_argument("--lambda_kd", type=float, default=1.0, help="Weight for combined KD loss (teacher → sparse+dense). RECOMMENDED: use only this")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin parameter (kept for compatibility, not used in KL loss)")
    parser.add_argument("--kd_temperature", type=float, default=1.0, help="Temperature for KL divergence in HybridDeepImpact")
    
    # Negative sampling for HybridDeepImpact
    parser.add_argument("--num_hard_negatives", type=int, default=8, help="Total number of hard negatives to sample for HybridDeepImpact (default: 8)")
    parser.add_argument("--top_k_hard", type=int, default=4, help="Number of top hard negatives to always include for HybridDeepImpact (default: 4)")

    # required for distillation loss with Margin MSE
    parser.add_argument("--qrels_path", type=Path, default=None, help="Path to the qrels file")

    args = parser.parse_args()

    assert not (args.distil_mse and args.distil_kl), "Cannot use both distillation losses at the same time"
    assert not (args.distil_mse and not args.qrels_path), "qrels_path is required for distillation loss with Margin MSE"
    assert not (args.hybrid and args.pairwise), "Cannot use hybrid and pairwise at the same time"
    assert not (args.hybrid and args.cross_encoder), "Cannot use hybrid and cross_encoder at the same time"
    assert not (args.hybrid and args.dense), "Cannot use hybrid and dense at the same time"
    assert not (args.hybrid and args.pairwise), "Cannot use hybrid and pairwise at the same time"
    assert not (args.hybrid and args.distil_mse), "Cannot use hybrid and distil_mse at the same time"
    assert not (args.hybrid and args.distil_kl), "Cannot use hybrid and distil_kl at the same time"

    # pass all argparse arguments to run() as kwargs
    run(**vars(args))
