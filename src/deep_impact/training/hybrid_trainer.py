import torch
import torch.nn.functional as F
from .trainer import Trainer
from .distil_trainer import DistilMarginMSE, DistilKLLoss, DistilTrainer


class HybridTrainer(Trainer):

    def __init__(
        self,
        model,
        optimizer,
        train_data,
        checkpoint_dir,
        batch_size,
        save_every,
        save_best=True,
        seed=42,
        gradient_accumulation_steps=1,
        evaluator=None,
        eval_every=500,
        lambda_di=1.0,
        lambda_li=1.0,
        lambda_kd=1.0,
        margin=0.2,
        temperature=1.0,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        self.lambda_di = lambda_di
        self.lambda_li = lambda_li
        self.lambda_kd = lambda_kd
        self.margin = margin
        self.temperature = temperature
        self.reset_tracking()

    def reset_tracking(self):
        self.sparse_loss_sum = 0
        self.dense_loss_sum = 0
        self.kd_loss_sum = 0
        self.sparse_loss_count = 0
        self.dense_loss_count = 0
        self.kd_loss_count = 0
        self.latest_score_stats = {}


    def compute_dense_and_combined_scores(
        self,
        q_dense,          # [B, Lq, dim]
        d_dense,          # [B, D, Ld, dim]
        q_dense_mask,     # [B, Lq]
        d_dense_mask,     # [B, D, Ld]
        sparse_scores,    # [B, D]
    ):
        """
        Compute dense_scores and combined_scores using model parameters:
        - model.temperature
        - model.log_sparse_weight
        - model.log_dense_weight
        """
        # Access model attributes (handle DDP wrapping)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        temp = model.temperature  # buffer or parameter
        B, D, Ld, dim = d_dense.shape
        Lq = q_dense.shape[1]

        # 1. Expand query embeddings for broadcasting
        q = q_dense.unsqueeze(1)         # [B, 1, Lq, dim]
        d = d_dense                      # [B, D, Ld, dim]

        # 2. Compute batched similarity with temperature
        # Cast temperature to match dense dtype (important for mixed precision)
        temp = temp.to(dtype=d_dense.dtype)
        sim = torch.matmul(q, d.transpose(2, 3)) / temp  # [B, D, Lq, Ld]

        # 3. Apply masks
        sim = sim.masked_fill(~d_dense_mask.unsqueeze(2).bool(), -1e4)
        sim = sim.masked_fill(~q_dense_mask.unsqueeze(1).unsqueeze(-1).bool(), -1e4)

        # 4. MaxSim over document tokens
        max_sim, _ = sim.max(dim=3)      # [B, D, Lq]

        # 5. Zero out padding query tokens
        max_sim = max_sim.masked_fill(~q_dense_mask.unsqueeze(1).bool(), 0.0)

        # 6. Sum over query dimension
        dense_scores = max_sim.sum(dim=2)   # [B, D]

        # 7. Combine sparse + dense with learnable weights
        weights = F.softmax(torch.stack([model.log_sparse_weight, model.log_dense_weight]), dim=0)
        ws = weights[0]
        wd = weights[1]
        # Cast weights to match score dtype (important for mixed precision)
        ws = ws.to(dtype=sparse_scores.dtype)
        wd = wd.to(dtype=dense_scores.dtype)
        
        combined_scores = ws * sparse_scores + wd * dense_scores

        return dense_scores, combined_scores

    # ===========================================================
    # Forward passes using encoded_list (consistent with other trainers)
    # ===========================================================

    def get_output_scores(self, batch):
        """
        Compute sparse, dense, and combined scores using encoded_list format.
        
        Args:
            batch: Dictionary with:
                - encoded_list: List of encoded documents with expansion tokens
                - masks: Sparse masks for overlapping terms [B*D, L, 1]
                - doc_dense_masks: Dense masks for expansion tokens [B*D, L, 1]
                - query_dense_masks: Dense masks for query tokens [B, L, 1]
                - num_docs_per_query: Number of documents per query
                
        Returns:
            Tuple of (sparse_scores, dense_scores, combined_scores, teacher_scores)
            Each tensor has shape [B, D] where B=batch_size, D=num_docs_per_query
        """
        device = self.device

        D = batch['num_docs_per_query']
        num_total = len(batch['encoded_list'])
        B = num_total // D

        # ===== Encode documents + queries TOGETHER (single forward pass for DDP) =====
        d_input_ids, d_attention_mask, d_type_ids = self.get_input_tensors(batch['encoded_list'])
        q_input_ids, q_attention_mask, q_type_ids = self.get_input_tensors(batch['query_encoded_list'])

        combined_input_ids = torch.cat([d_input_ids, q_input_ids], dim=0)
        combined_attention_mask = torch.cat([d_attention_mask, q_attention_mask], dim=0)
        combined_type_ids = torch.cat([d_type_ids, q_type_ids], dim=0)

        combined_sparse, combined_dense = self.model(
            combined_input_ids,
            combined_attention_mask,
            combined_type_ids,
            return_dense_embeddings=True,
        )

        L = combined_sparse.size(1)
        num_docs_total = B * D
        num_queries = B

        d_sparse, _ = torch.split(combined_sparse, [num_docs_total, num_queries], dim=0)
        d_dense, q_dense = torch.split(combined_dense, [num_docs_total, num_queries], dim=0)

        d_sparse = d_sparse.view(B, D, L, 1)
        d_dense  = d_dense.view(B, D, L, -1)
        q_dense = q_dense.view(B, L, -1)

        # masks - ensure they're float tensors for mixed precision compatibility
        sparse_masks = batch['masks'].to(device).float().view(B, D, L, 1)
        d_dense_mask = batch['doc_dense_masks'].to(device).float().view(B, D, L, 1).squeeze(-1)  # [B, D, L]
        q_dense_mask = batch['query_dense_masks'].to(device).squeeze(-1)  # Already float from collate

        # ===== 3. Sparse scores (unchanged) =====
        sparse_scores = (sparse_masks * d_sparse).sum(dim=2).squeeze(-1)  # [B, D]

        # ===== 4. Dense + combined scores INSIDE MODEL =====
        # IMPORTANT: call the helper we just added
        # Note: self.model is DDP-wrapped; we need the underlying module.
        dense_scores, combined_scores = self.compute_dense_and_combined_scores(
            q_dense=q_dense,
            d_dense=d_dense,
            q_dense_mask=q_dense_mask,
            d_dense_mask=d_dense_mask,
            sparse_scores=sparse_scores,
        )

        teacher_scores = batch["scores"].view(B, D).to(device)

        return sparse_scores, dense_scores, combined_scores, teacher_scores          



    # ===========================================================
    # Loss functions (all using KL divergence for distillation)
    # ===========================================================
    def compute_sparse_kd_loss(self, teacher_scores, sparse_scores):
        """
        KL divergence loss for sparse scores only.
        
        Distills teacher knowledge into the sparse (lexical) component.
        Does NOT require positive/negative distinction.
        
        Args:
            teacher_scores: Teacher scores from cross-encoder [B, D]
            sparse_scores: Student sparse scores [B, D]
            
        Returns:
            KL divergence loss for sparse component
        """
        teacher_p = F.softmax(teacher_scores / self.temperature, dim=1)
        sparse_log = F.log_softmax(sparse_scores / self.temperature, dim=1)
        
        # KL(teacher || sparse)
        loss = -(teacher_p * sparse_log).sum(dim=1).mean()
        return loss

    def compute_dense_kd_loss(self, teacher_scores, dense_scores):
        """
        KL divergence loss for dense scores only.
        
        Distills teacher knowledge into the dense (semantic) component.
        Does NOT require positive/negative distinction.
        
        Args:
            teacher_scores: Teacher scores from cross-encoder [B, D]
            dense_scores: Student dense scores [B, D]
            
        Returns:
            KL divergence loss for dense component
        """
        teacher_p = F.softmax(teacher_scores / self.temperature, dim=1)
        dense_log = F.log_softmax(dense_scores / self.temperature, dim=1)
        
        # KL(teacher || dense)
        loss = -(teacher_p * dense_log).sum(dim=1).mean()
        return loss

    def compute_combined_kd_loss(self, teacher_scores, sparse_scores, dense_scores):
        """
        KL divergence loss for the combined (sparse + dense) student scores.
        
        This distills teacher knowledge into the FINAL combined output of the model.
        The student's combined score (sparse + dense) is matched to the teacher.
        
        This is cleaner than separately distilling sparse and dense components,
        as it directly trains the final retrieval score.
        
        Args:
            teacher_scores: Teacher scores from cross-encoder [B, D]
            sparse_scores: Student sparse scores [B, D]
            dense_scores: Student dense scores [B, D]
            
        Returns:
            KL divergence loss for combined student scores
        """
        # Combine sparse and dense scores (simple addition)
        combined_student_scores = sparse_scores + dense_scores
        
        # Compute KL divergence between teacher and combined student
        teacher_p = F.softmax(teacher_scores / self.temperature, dim=1)
        combined_log = F.log_softmax(combined_student_scores / self.temperature, dim=1)
        
        # KL(teacher || combined_student)
        loss = -(teacher_p * combined_log).sum(dim=1).mean()
        
        return loss

    # ===========================================================
    def evaluate_loss(self, outputs, batch):
        """
        Compute combined loss with configurable weights.
        
        All losses use KL divergence for knowledge distillation:
        - lambda_di: Weight for sparse component KD loss (teacher → sparse only)
        - lambda_li: Weight for dense component KD loss (teacher → dense only)
        - lambda_kd: Weight for combined KD loss (teacher → sparse+dense combined)
        
        RECOMMENDED: Use ONLY lambda_kd=1.0 for most cases.
        This distills the teacher into the FINAL combined output (sparse+dense),
        which is what you actually use for retrieval.
        
        Alternative usage (if you want to emphasize individual components):
        - lambda_di=0.3, lambda_li=0.3, lambda_kd=0.4
        
        Note: lambda_kd uses (sparse + dense) as the student score, which is
        different from (lambda_di + lambda_li) which treats them separately.
        """
        sparse, dense, combined, teacher = outputs
        
        # Use teacher scores from batch (cross-encoder scores)
        teacher_scores = batch['scores'].view(sparse.shape).to(self.device)

        # Compute losses only if their weights are non-zero
        if self.lambda_di > 0:
            loss_di = self.compute_sparse_kd_loss(teacher_scores, sparse)
        else:
            loss_di = torch.tensor(0.0, device=self.device)
            
        if self.lambda_li > 0:
            loss_li = self.compute_dense_kd_loss(teacher_scores, dense)
        else:
            loss_li = torch.tensor(0.0, device=self.device)
            
        if self.lambda_kd > 0:
            loss_kd = self.compute_combined_kd_loss(teacher_scores, sparse, dense)
        else:
            loss_kd = torch.tensor(0.0, device=self.device)

        loss = (
            self.lambda_di * loss_di +
            self.lambda_li * loss_li +
            self.lambda_kd * loss_kd
        )

        # Track (only if non-zero)
        if self.lambda_di > 0:
            self.sparse_loss_sum += loss_di.item()
            self.sparse_loss_count += 1
        if self.lambda_li > 0:
            self.dense_loss_sum += loss_li.item()
            self.dense_loss_count += 1
        if self.lambda_kd > 0:
            self.kd_loss_sum += loss_kd.item()
            self.kd_loss_count += 1

        with torch.no_grad():
            self.latest_score_stats = {
                "sparse_mean": sparse.mean().item(),
                "sparse_std": sparse.std(unbiased=False).item() if sparse.numel() > 1 else 0.0,
                "dense_mean": dense.mean().item(),
                "dense_std": dense.std(unbiased=False).item() if dense.numel() > 1 else 0.0,
                "combined_mean": combined.mean().item(),
                "combined_std": combined.std(unbiased=False).item() if combined.numel() > 1 else 0.0,
            }

        return loss

    def train(self):
        self.reset_tracking()
        super().train()

    def _log_batch(self, idx, loss):
        if idx % 10 == 0:
            di = (self.sparse_loss_sum / self.sparse_loss_count) if self.sparse_loss_count else 0.0
            li = (self.dense_loss_sum / self.dense_loss_count) if self.dense_loss_count else 0.0
            kd = (self.kd_loss_sum / self.kd_loss_count) if self.kd_loss_count else 0.0
            print(f"[Batch {idx}] Loss={loss:.4f} DI={di:.4f} LI={li:.4f} KD={kd:.4f}")
            if self.latest_score_stats:
                stats = self.latest_score_stats
                print(
                    "           Scores | "
                    f"Sparse μ={stats['sparse_mean']:.4f} σ={stats['sparse_std']:.4f} "
                    f"| Dense μ={stats['dense_mean']:.4f} σ={stats['dense_std']:.4f} "
                    f"| Combined μ={stats['combined_mean']:.4f} σ={stats['combined_std']:.4f}"
                )
