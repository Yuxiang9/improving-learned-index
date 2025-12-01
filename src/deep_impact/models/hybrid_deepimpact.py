"""
Hybrid Deep Impact: Hybrid Sparse-Dense Retrieval with Late Interaction

This model implements a hybrid approach inspired by the MetaEmbed paper:
- Sparse component: Traditional DeepImpact impact scores for lexical matching
- Dense component: Special expansion tokens with dense embeddings for late interaction (MaxSim)
- Joint training: Both components trained together with contrastive learning

The key difference from standard DeepImpact:
1. Expansion tokens produce dense embeddings (not scalar scores)
2. Late interaction scoring using MaxSim (like ColBERT) on expansion tokens
3. Combined sparse + dense score for final ranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Set, List, Optional
from .original import DeepImpact
from collections import namedtuple
EncodingLike = namedtuple('EncodingLike', ['ids', 'attention_mask', 'type_ids'])

class ExpansionEmbedding(nn.Module):
    """
    Produces ColBERT-style dense embeddings for expansion-token slots.
    
    Components:
      1. Dense projection MLP → transforms BERT hidden states
      2. Slot position embeddings → give each expansion slot a unique identity
      3. Transformer refinement → allows cross-slot interactions and specialization
    
    Input:
      hidden_states: [B, L, hidden_dim]
      input_ids: [B, L]
      exp_id: tokenizer ID of the single expansion token "[EXP]"
    
    Output:
      dense_embeddings: [B, L, dense_dim]  (normalized)
    """
    def __init__(self, hidden_dim, dense_dim, num_slots, num_layers=2):
        super().__init__()
        
        # 1. Dense Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, dense_dim),
        )
        
        # 2. Slot position embedding (colbert-style identity)
        self.slot_pos_emb = nn.Embedding(num_slots, dense_dim)

        # 3. Transformer refinement (cross-slot + contextual smoothing)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dense_dim,
            nhead=8,
            dim_feedforward=dense_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.num_slots = num_slots
        self.dense_dim = dense_dim


    def forward(self, hidden_states, input_ids, exp_id):
        """
        hidden_states: [B, L, H]
        input_ids:     [B, L]
        """
        B, L, _ = hidden_states.shape
        
        # 1. Dense MLP projection
        dense = self.projection(hidden_states)     # [B, L, dim]

        # 2. Identify expansion token positions
        exp_mask = (input_ids == exp_id)           # [B, L]
        
        # 3. Add slot positional embeddings WITHOUT in-place operations
        # Create a separate tensor to accumulate positional additions
        pos_additions = torch.zeros_like(dense)
        
        for b in range(B):
            pos = exp_mask[b].nonzero(as_tuple=True)[0]   # e.g., [0,1,2,...,K-1]
            K = pos.size(0)
            if K > 0:
                slot_emb = self.slot_pos_emb(
                    torch.arange(K, device=dense.device)
                )                                   # [K, dim]
                # Cast to same dtype as dense (important for mixed precision training)
                slot_emb = slot_emb.to(dtype=dense.dtype)
                # Safe: we're only writing to pos_additions, not dense
                pos_additions[b, pos] = slot_emb
        
        # Add positional embeddings (not in-place!)
        dense = dense + pos_additions

        # 4. Performer / Transformer refinement
        dense = self.transformer(dense)             # [B, L, dim]
        
        # 5. L2 normalize for cosine scoring
        dense = F.normalize(dense, p=2, dim=-1)

        return dense

class HybridDeepImpact(DeepImpact):
    """
    Hybrid sparse-dense retrieval model combining:
    1. Sparse impact scores for document terms (lexical matching)
    2. Dense late-interaction for expansion tokens (semantic matching)
    
    Architecture:
    - BERT backbone (shared)
    - Sparse head: Linear + ReLU → scalar impact scores for all tokens
    - Dense head: Linear projection → dense embeddings ONLY for expansion tokens
    
    Scoring:
    - Sparse score: sum of impact scores for matching query terms
    - Dense score: MaxSim late interaction on expansion token embeddings
    - Final score: weighted combination of sparse + dense
    """
    
    exp_token = "[EXP]"
    exp_id = None     # filled after tokenizer.add_tokens
    num_expansion_slots = None   # how many [EXP] we repeat

    @classmethod
    def configure_expansion_token(cls, num_slots: int):
        """
        Setup the single expansion token [EXP] and remember that we will
        repeat it `num_slots` times after each document.
        """
        cls.num_expansion_slots = num_slots
        if cls.exp_token in cls.tokenizer.get_added_vocab():
            cls.exp_id = cls.tokenizer.convert_tokens_to_ids(cls.exp_token)
            return


        # Add token if missing
        added = cls.tokenizer.add_special_tokens({"additional_special_tokens": [cls.exp_token]})
        cls.exp_id = cls.tokenizer.convert_tokens_to_ids(cls.exp_token)

        # Record id
        cls.exp_id = cls.tokenizer.convert_tokens_to_ids(cls.exp_token)

    def __init__(self, config, dense_dim: int = 128, num_expansion_tokens: int = 64, 
                 sparse_weight: float = 0.5, dense_weight: float = 0.5,
                 learn_weights: bool = True):
        """
        Initialize Hybrid Deep Impact model.
        
        Args:
            config: BERT configuration
            dense_dim: Dimensionality of dense embeddings for expansion tokens (default: 128)
            num_expansion_tokens: Number of special expansion tokens to add (default: 64)
            sparse_weight: Weight for sparse component (default: 0.5)
            dense_weight: Weight for dense component (default: 0.5)
            learn_weights: If True, weights are learnable parameters. If False, fixed. (default: True)
        """
        # First, update the class-level expansion tokens before calling super().__init__
        # This ensures the tokenizer has the correct number of expansion tokens
        # Note: set_expansion_tokens should have been called before __init__ in train.py
        # But we update the list here for consistency
        self.__class__.configure_expansion_token(num_expansion_tokens)
        
        # Initialize parent (this will call BertPreTrainedModel.__init__)
        super().__init__(config)

        # Ensure token embeddings are resized to the new vocabulary size
        # tokenizer.vocab_size does not include newly added tokens; use len(...)
        new_vocab_size = len(self.tokenizer)
        self.resize_token_embeddings(new_vocab_size)

        # Initialize expansion token embeddings properly (important!)
        with torch.no_grad():
            w = self.bert.embeddings.word_embeddings.weight
            new = self.num_expansion_slots
            w[-new:].normal_(mean=0.0, std=0.02)
        
        # Store configuration
        self.dense_dim = dense_dim
        self.num_expansion_slots = num_expansion_tokens
        self.learn_weights = learn_weights
        
        # Sparse head: produces scalar impact scores for ALL tokens (standard DeepImpact)
        # This is already initialized by parent as self.impact_score_encoder
        
        # Dense head: produces dense embeddings ONLY for expansion tokens
        # We'll extract these embeddings and use them for late interaction
        self.expansion_embedding = ExpansionEmbedding(
            hidden_dim=config.hidden_size,
            dense_dim=dense_dim,
            num_slots=num_expansion_tokens,
            num_layers=2,   # recommended
        )
        
        # Combination weights: learnable OR fixed
        if learn_weights:
            # Learnable weights (trained via gradient descent)
            # Using log space for numerical stability during training
            self.log_sparse_weight = nn.Parameter(torch.tensor(sparse_weight).log())
            self.log_dense_weight = nn.Parameter(torch.tensor(dense_weight).log())
        else:
            # Fixed weights (not trainable) - avoids DDP issues entirely!
            # Register as buffers so they're part of state_dict but not parameters
            self.register_buffer('log_sparse_weight', torch.tensor(sparse_weight).log())
            self.register_buffer('log_dense_weight', torch.tensor(dense_weight).log())
        
        # Temperature parameter for late interaction (like ColBERT)
        # This scales the similarity scores
        # Always learnable for now (can make this optional too if needed)
        # self.temperature = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("temperature", torch.tensor(1.0))
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        return_dense_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass producing both sparse scores and (optionally) dense embeddings.
        
        Args:
            input_ids: Batch of input ids [batch_size, seq_len]
            attention_mask: Batch of attention masks [batch_size, seq_len]
            token_type_ids: Batch of token type ids [batch_size, seq_len]
            return_dense_embeddings: If True, return dense embeddings for expansion tokens
            
        Returns:
            - sparse_scores: Impact scores for all tokens [batch_size, seq_len, 1]
            - dense_embeddings: (if return_dense_embeddings=True) 
                               Dense embeddings [batch_size, seq_len, dense_dim]
        """
        # Get BERT representations
        bert_output = self._get_bert_output(input_ids, attention_mask, token_type_ids)
        last_hidden_state = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Sparse scores: keep computation in fp32 to avoid NaNs in mixed precision
        sparse_input = last_hidden_state
        if sparse_input.dtype != torch.float32:
            sparse_input = sparse_input.float()
        sparse_scores = self.impact_score_encoder(sparse_input)  # [batch_size, seq_len, 1]
        
        # Dense embeddings: apply projection and normalize for dot product similarity
        if return_dense_embeddings:
            dense_embeddings = self.expansion_embedding(last_hidden_state, input_ids, self.exp_id)  # [batch_size, seq_len, dense_dim]
            return sparse_scores, dense_embeddings
        
        return sparse_scores, None
    
    def compute_late_interaction_score(
        self,
        query_embeddings: torch.Tensor,   # [B, Lq, dim]
        doc_embeddings: torch.Tensor,     # [B, Ld, dim]
        query_mask: torch.Tensor,         # [B, Lq]
        doc_mask: torch.Tensor,           # [B, Ld]
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if temperature is None:
            temperature = self.temperature.item()

        # Ensure masks are boolean
        q_mask = query_mask.bool()     # [B, Lq]
        d_mask = doc_mask.bool()       # [B, Ld]

        # Fully batched similarity
        # sim[b] = Q[b] @ D[b].T → shape [B, Lq, Ld]
        sim = torch.matmul(
            query_embeddings,                      # [B, Lq, dim]
            doc_embeddings.transpose(1, 2)         # [B, dim, Ld]
        ) / temperature

        # Mask out invalid doc tokens
        # d_mask → [B, 1, Ld]
        sim = sim.masked_fill(~d_mask.unsqueeze(1), -1e4)

        # MaxSim: max over document token axis
        # max_sim[b, q] = max_j sim[b, q, j]
        max_sim, _ = sim.max(dim=2)    # [B, Lq]

        # Mask invalid query tokens
        max_sim = max_sim.masked_fill(~q_mask, 0.0)

        # Sum over query tokens
        # score[b] = Σ_q max_sim[b, q]
        scores = max_sim.sum(dim=1)    # [B]

        return scores
    
    def get_combined_scores(
        self,
        query_sparse_scores: torch.Tensor,
        doc_sparse_scores: torch.Tensor,
        query_dense_embeddings: torch.Tensor,
        doc_dense_embeddings: torch.Tensor,
        query_sparse_mask: torch.Tensor,
        doc_sparse_mask: torch.Tensor,
        query_dense_mask: torch.Tensor,
        doc_dense_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combine sparse and dense scores with learned weights.
        
        Args:
            query_sparse_scores: Query sparse scores [batch_size, seq_len, 1]
            doc_sparse_scores: Document sparse scores [batch_size, seq_len, 1]
            query_dense_embeddings: Query dense embeddings [batch_size, seq_len, dense_dim]
            doc_dense_embeddings: Document dense embeddings [batch_size, seq_len, dense_dim]
            query_sparse_mask: Mask for query terms in document [batch_size, seq_len, 1]
            doc_sparse_mask: Not used (keeping for interface compatibility)
            query_dense_mask: Mask for query expansion tokens [batch_size, seq_len]
            doc_dense_mask: Mask for document expansion tokens [batch_size, seq_len]
            
        Returns:
            Tuple of (sparse_score, dense_score, combined_score) each [batch_size]
        """
        # Sparse score: sum of impact scores for matching query terms in document
        # This is the standard DeepImpact scoring
        sparse_score = (query_sparse_mask * doc_sparse_scores).sum(dim=1).squeeze(-1)
        
        # Dense score: MaxSim late interaction on expansion tokens
        dense_score = self.compute_late_interaction_score(
            query_dense_embeddings,
            doc_dense_embeddings,
            query_dense_mask,
            doc_dense_mask,
        )
        
        # Combine with learned weights (using softmax for normalization)
        weights = F.softmax(torch.stack([self.log_sparse_weight, self.log_dense_weight]), dim=0)
        sparse_weight = weights[0]
        dense_weight = weights[1]
        
        combined_score = sparse_weight * sparse_score + dense_weight * dense_score
        
        return sparse_score, dense_score, combined_score
    
    @classmethod
    def load(cls, checkpoint_path: Optional[str] = None, dense_dim: int = 128, 
             num_expansion_tokens: int = 64, learn_weights: bool = True):
        """
        Load model with proper initialization of expansion tokens.
        
        Args:
            checkpoint_path: Path to checkpoint (if None, loads pretrained base)
            dense_dim: Dimensionality of dense embeddings
            num_expansion_tokens: Number of expansion tokens to use
            learn_weights: If True, weights are learnable. If False, fixed.
            
        Returns:
            Loaded model
        """
        from transformers import AutoConfig
        import os
        from src.utils.checkpoint import ModelCheckpoint
        
        # Update class-level expansion tokens BEFORE creating model
        # Use the parent class method to properly update tokenizer
        cls.set_expansion_tokens(num_expansion_tokens)
        
        # Load configuration
        config = AutoConfig.from_pretrained('Luyu/co-condenser-marco')
        
        # Create model
        model = cls(config, dense_dim=dense_dim, num_expansion_tokens=num_expansion_tokens,
                   learn_weights=learn_weights)
        
        # Resize token embeddings to account for new expansion tokens
        # cls.tokenizer.enable_truncation(max_length=cls.max_length, strategy='longest_first')
        # cls.tokenizer.enable_padding(length=cls.max_length)
        cls.tokenizer.model_max_length = cls.max_length
        vocab_size = cls.tokenizer.vocab_size
        
        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ModelCheckpoint.load(model=model, last_checkpoint_path=checkpoint_path)
        
        return model
    
    @classmethod
    def process_query_and_document(
        cls, 
        query: str, 
        document: str, 
        max_length: Optional[int] = None
    ) -> Tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process query and document with expansion tokens for HybridDeepImpact training.
        
        This method:
        1. Tokenizes the document
        2. Appends expansion tokens BEFORE padding
        3. Creates masks for sparse and dense components
        
        Args:
            query: Query string
            document: Document string
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (encoded_document, sparse_mask, query_dense_mask, doc_dense_mask)
            - encoded_document: Tokenized document with expansion tokens
            - sparse_mask: Mask for overlapping query terms (for sparse scoring)
            - query_dense_mask: Not used in collate, but kept for consistency
            - doc_dense_mask: Mask for expansion tokens (for dense scoring)
        """
        if max_length is None:
            max_length = cls.max_length
        
        # Process query to get query terms
        query_terms = cls.process_query(query)
        
        # Tokenize document without padding/truncation first
        document_normalized = cls.tokenizer.backend_tokenizer.normalizer.normalize_str(document)
        document_terms = [x[0] for x in cls.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(document_normalized)]
        # Use backend_tokenizer.encode which returns an Encoding object
        encoded = cls.tokenizer.backend_tokenizer.encode(document_terms, is_pretokenized=True)
        
        # Calculate how much space we have for document + expansion tokens
        num_exp = cls.num_expansion_slots if cls.num_expansion_slots is not None else 64
        max_doc_length = max_length - num_exp
        
        # Truncate document tokens if needed (keeping CLS token)
        ids = encoded.ids[:max_doc_length]
        mask = encoded.attention_mask[:max_doc_length]
        type_ids = encoded.type_ids[:max_doc_length]
        
        # Map only surviving terms + only up to max_doc_length
        term_to_token_index = {}
        counter = 0
        for tok_i, token in enumerate(encoded.tokens[1:], start=1):   # skip CLS
            if tok_i >= max_doc_length:
                break
            if token.startswith("##"):
                continue
            term_to_token_index[counter] = tok_i
            counter += 1

        # Filter based on surviving tokens only
        filtered_term_to_token_index = {}
        for i, term in enumerate(document_terms):
            if i in term_to_token_index and term not in cls.punctuation:
                if term not in filtered_term_to_token_index:
                    filtered_term_to_token_index[term] = term_to_token_index[i]        

        # Append expansion tokens
        exp_id = cls.exp_id
        if exp_id is None:
            raise RuntimeError("Expansion tokens not configured! Call configure_expansion_token first.")
        
        ids = ids + [exp_id] * num_exp
        mask = mask + [1] * num_exp
        type_ids = type_ids + [0] * num_exp
        
        # Pad to max_length
        if len(ids) < max_length:
            pad_len = max_length - len(ids)
            pad_id = cls.tokenizer.pad_token_id
            ids = ids + [pad_id] * pad_len
            mask = mask + [0] * pad_len
            type_ids = type_ids + [0] * pad_len
        
        # Create a simple encoding-like object
        # (tokenizers.Encoding constructor changed, can't use keyword args directly)
        from collections import namedtuple
        EncodingLike = namedtuple('EncodingLike', ['ids', 'attention_mask', 'type_ids'])
        encoded_with_exp = EncodingLike(
            ids=ids,
            attention_mask=mask,
            type_ids=type_ids,
        )
        
        # Create masks
        import numpy as np
        
        # 1. Sparse mask: overlapping query terms (standard DeepImpact mask)
        sparse_mask = np.zeros(max_length, dtype=bool)
        overlapping_indices = [idx for term, idx in filtered_term_to_token_index.items() if term in query_terms]
        sparse_mask[overlapping_indices] = True
        
        # 2. Query dense mask: placeholder (will be computed in collate for actual query)
        query_dense_mask = np.ones(max_length, dtype=bool)  # Placeholder
        
        # 3. Document dense mask: only expansion token positions
        doc_dense_mask = np.zeros(max_length, dtype=bool)
        exp_start_idx = max_doc_length
        exp_end_idx = max_doc_length + num_exp
        doc_dense_mask[exp_start_idx:exp_end_idx] = True
        
        return (
            encoded_with_exp,
            torch.from_numpy(sparse_mask),
            torch.from_numpy(query_dense_mask),
            torch.from_numpy(doc_dense_mask)
        )
    
    @classmethod
    def get_expansion_token_mask(
        cls,
        term_to_token_index: Dict[str, int],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create a mask specifically for expansion tokens.
        
        Args:
            term_to_token_index: Mapping from terms to token indices
            max_length: Maximum sequence length
            
        Returns:
            Boolean mask with True for expansion token positions
        """
        if max_length is None:
            max_length = cls.max_length
        
        import numpy as np
        mask = np.zeros(max_length, dtype=bool)
        
        # Mark only expansion token positions
        expansion_indices = [
            idx for term, idx in term_to_token_index.items()
            if term in cls.expansion_tokens
        ]
        mask[expansion_indices] = True
        
        return torch.from_numpy(mask)
    
    def get_impact_scores_batch(self, documents: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Get impact scores for inference (uses only sparse component for efficiency).
        For full sparse+dense retrieval, use the appropriate retrieval pipeline.
        
        Args:
            documents: List of document strings
            
        Returns:
            List of lists of (term, impact_score) tuples
        """
        # For inference, we only use sparse scores (dense requires query)
        encoded_docs = []
        term_to_token_maps = []
        for doc in documents:
            encoded, term_map = self.process_document(doc)
            encoded_docs.append(encoded)
            term_to_token_maps.append(term_map)

        # Create batched tensors
        input_ids = torch.tensor([enc.ids for enc in encoded_docs], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([enc.attention_mask for enc in encoded_docs], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([enc.type_ids for enc in encoded_docs], dtype=torch.long).to(self.device)

        # Get sparse scores only
        with torch.no_grad():
            sparse_scores, _ = self(input_ids, attention_mask, token_type_ids, return_dense_embeddings=False)

        # Compute impact scores for all documents
        return self.compute_term_impacts(term_to_token_maps, sparse_scores)
