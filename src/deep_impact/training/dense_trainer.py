import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer

from typing import Dict, List, Tuple

from .trainer import Trainer
from src.deep_impact.models import DeepImpact
from .distil_trainer import DistilTrainer

class DenseBiEncoder(nn.Module):
    """A lightweight dense bi-encoder using a pretrained BERT-style backbone.

    The encoder maintains two independent towers (query / document). After the
    backbone the hidden representation of the [CLS] token is L2-normalised and
    used as the embedding. The dot-product of query and document embeddings is
    treated as the relevance score.
    """

    def __init__(self, model_name: str = "bert-base-uncased", proj_dim: int = 768):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

        hidden = self.query_encoder.config.hidden_size
        # Optional projection (helps when hidden != proj_dim or to add extra capacity).
        if proj_dim != hidden:
            self.query_proj = nn.Linear(hidden, proj_dim)
            self.doc_proj = nn.Linear(hidden, proj_dim)
        else:
            self.query_proj = self.doc_proj = None

    def _encode(self, encoder: AutoModel, proj: nn.Linear, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = out.last_hidden_state[:, 0, :]  # CLS token
        if proj is not None:
            emb = proj(emb)
        return F.normalize(emb, dim=1)

    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self._encode(self.query_encoder, self.query_proj, input_ids, attention_mask)

    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self._encode(self.doc_encoder, self.doc_proj, input_ids, attention_mask)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns a relevance *score* (dot-product) for every query / document pair in the batch.

        The batch is expected to be organised such that `query_input_ids[i]` is
        paired with `doc_input_ids[i]`.
        """
        q_emb = self.encode_query(query_input_ids, query_attention_mask)
        d_emb = self.encode_doc(doc_input_ids, doc_attention_mask)
        return (q_emb * d_emb).sum(dim=1)  # shape: (batch,)


# ---------------------------------------------------------------------------
# Collate-function to jointly prepare inputs for both sparse and dense models
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}

def _get_tokenizer(name: str) -> AutoTokenizer:
    if name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER_CACHE[name]


def dense_joint_collate_fn(
    batch: List[Tuple[str, str, str]],
    model_cls=DeepImpact,
    dense_tokenizer_name: str = "bert-base-uncased",
    max_length_sparse: int = 300,
    max_length_query: int = 32,
    max_length_doc: int = 256,
) -> Dict[str, torch.Tensor]:
    """Create a training batch for joint sparse (DeepImpact) + dense bi-encoder.

    The input *batch* is expected to be a list of triples (query, pos_doc,
    neg_doc).  For each triple we produce:
        • DeepImpact encodings (document only) + mask - twice (pos & neg)
        • Dense encodings for query and corresponding document (two pairs)
    The positive document ALWAYS comes first so that label = 0 when using
    CrossEntropyLoss.
    """

    # Containers for sparse inputs
    encoded_list, masks = [], []

    # Containers for dense inputs
    q_input_ids, q_attn, d_input_ids, d_attn = [], [], [], []

    dense_tok = _get_tokenizer(dense_tokenizer_name)

    for query, pos_doc, neg_doc in batch:
        # --- Sparse DeepImpact ------------------------------------------------
        enc_pos, mask_pos = model_cls.process_query_and_document(query, pos_doc, max_length=max_length_sparse)
        enc_neg, mask_neg = model_cls.process_query_and_document(query, neg_doc, max_length=max_length_sparse)

        encoded_list.extend([enc_pos, enc_neg])
        masks.extend([mask_pos, mask_neg])

        # --- Dense bi-encoder -------------------------------------------------
        # Encode query once (will be repeated for pos / neg documents)
        q_enc = dense_tok(
            query,
            truncation=True,
            padding="max_length",
            max_length=max_length_query,
            return_tensors="pt",
        )

        pos_enc = dense_tok(
            pos_doc,
            truncation=True,
            padding="max_length",
            max_length=max_length_doc,
            return_tensors="pt",
        )
        neg_enc = dense_tok(
            neg_doc,
            truncation=True,
            padding="max_length",
            max_length=max_length_doc,
            return_tensors="pt",
        )

        # Positive pair
        q_input_ids.append(q_enc["input_ids"].squeeze(0))
        q_attn.append(q_enc["attention_mask"].squeeze(0))
        d_input_ids.append(pos_enc["input_ids"].squeeze(0))
        d_attn.append(pos_enc["attention_mask"].squeeze(0))
        # Negative pair
        q_input_ids.append(q_enc["input_ids"].squeeze(0))
        q_attn.append(q_enc["attention_mask"].squeeze(0))
        d_input_ids.append(neg_enc["input_ids"].squeeze(0))
        d_attn.append(neg_enc["attention_mask"].squeeze(0))

    return {
        "encoded_list": encoded_list,
        "masks": torch.stack(masks, dim=0).unsqueeze(-1),
        # Dense inputs
        "query_input_ids": torch.stack(q_input_ids, dim=0),
        "query_attention_mask": torch.stack(q_attn, dim=0),
        "doc_input_ids": torch.stack(d_input_ids, dim=0),
        "doc_attention_mask": torch.stack(d_attn, dim=0),
    }


# ---------------------------------------------------------------------------
# Joint Trainer
# ---------------------------------------------------------------------------

class DenseTrainer(DistilTrainer):
    """Trainer that jointly optimises a sparse DeepImpact model and a dense bi-encoder.

    The final relevance score is obtained by *adding* the sparse impact score
    (after term masking) and the dense similarity score. The rest of the
    training loop - loss, checkpointing, evaluation - is inherited from the
    base `Trainer`.
    """

    def __init__(
        self,
        sparse_model: DistilTrainer,
        dense_model: DenseBiEncoder,
        optimizer: torch.optim.Optimizer,
        train_data,
        checkpoint_dir,
        batch_size: int,
        save_every: int,
        save_best: bool = True,
        seed: int = 42,
        gradient_accumulation_steps: int = 1,
        eval_every: int = 500,
        evaluator=None,
    ):
        # Pass the sparse model to the base Trainer – it will be wrapped in DDP
        super().__init__(
            model=sparse_model,
            optimizer=optimizer,
            train_data=train_data,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_every=eval_every,
            evaluator=evaluator,
        )

        # Keep an explicit reference to the underlying sparse model *module* so
        # that we can call it directly irrespective of DDP-wrapping.
        self.sparse_model = self.model.module  # type: DeepImpact

        # Setup dense model (with its own DDP wrapper) -------------------------------------------------
        self.dense_model = dense_model.to(self.gpu_id)
        self.dense_model = DDP(self.dense_model, device_ids=[self.gpu_id], find_unused_parameters=True)

        # Make sure optimiser updates *both* models. The user may have already
        # included the dense parameters, but if not – add a param group.
        dense_params_ids = {id(p) for p in self.dense_model.parameters()}
        already_present = any(id(p) in dense_params_ids for group in optimizer.param_groups for p in group["params"])
        if not already_present:
            self.optimizer.add_param_group({"params": self.dense_model.parameters()})

    # ---------------------------------------------------------------------
    # Overridden helpers
    # ---------------------------------------------------------------------

    def train(self):
        # Ensure both models are in training mode before delegating to base class
        self.dense_model.train()
        super().train()

    # We reuse the loss from base class – only the *scores* change
    def get_output_scores(self, batch):
        # ---------------- Sparse scores ----------------
        # Delegate to the implementation in the parent Trainer instead of duplicating logic.
        sparse_scores = super().get_output_scores(batch)

        # ---------------- Dense scores -----------------
        q_ids = batch["query_input_ids"].to(self.gpu_id)
        q_mask = batch["query_attention_mask"].to(self.gpu_id)
        d_ids = batch["doc_input_ids"].to(self.gpu_id)
        d_mask = batch["doc_attention_mask"].to(self.gpu_id)

        dense_pair_scores = self.dense_model(q_ids, q_mask, d_ids, d_mask)  # shape: (batch*2,)
        dense_scores = dense_pair_scores.view(self.batch_size, -1)

        # ---------------- Aggregate --------------------
        return sparse_scores + dense_scores
