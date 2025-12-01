import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiEncoder(nn.Module):
    """
    Bi-encoder model for query and document encoding with separate encoders using BGE-M3
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", hidden_dim: int = 1024):
        super().__init__()
        # Use BGE-M3 as the backbone - SOTA dense retrieval model
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        # BGE-M3 has 1024 hidden dimensions by default
        encoder_hidden_size = getattr(self.query_encoder.config, 'hidden_size', 1024)
        
        # Projection layers to normalize embedding dimensions
        self.query_projection = nn.Linear(encoder_hidden_size, hidden_dim)
        self.doc_projection = nn.Linear(encoder_hidden_size, hidden_dim)
        
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode query text"""
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        query_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.query_projection(query_embedding), dim=1)
    
    def encode_doc(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode document text"""
        outputs = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        doc_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.doc_projection(doc_embedding), dim=1)
    
    def forward(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                doc_input_ids: torch.Tensor, doc_attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for both query and document"""
        query_emb = self.encode_query(query_input_ids, query_attention_mask)
        doc_emb = self.encode_doc(doc_input_ids, doc_attention_mask)
        
        # Compute similarity scores
        scores = torch.matmul(query_emb, doc_emb.transpose(0, 1))
        
        return {
            'query_embeddings': query_emb,
            'doc_embeddings': doc_emb,
            'scores': scores
        }

class TeacherModel(nn.Module):
    """
    Teacher model - using BGE-Large as a larger, more complex cross-encoder teacher
    """
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_hidden_size = getattr(self.encoder.config, 'hidden_size', 1024)
        self.classifier = nn.Linear(encoder_hidden_size, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for teacher model (cross-encoder style)"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)

class QueryDocDataset(Dataset):
    """Dataset for query-document pairs"""
    
    def __init__(self, queries: List[str], documents: List[str], labels: List[float],
                 tokenizer: AutoTokenizer, max_length: int = 512):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]
        label = self.labels[idx]
        
        # Tokenize query
        query_encoded = self.tokenizer(
            query, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length//2,
            return_tensors='pt'
        )
        
        # Tokenize document
        doc_encoded = self.tokenizer(
            document, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length//2,
            return_tensors='pt'
        )
        
        # For teacher model - concatenate query and document
        combined_text = f"{query} [SEP] {document}"
        combined_encoded = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoded['input_ids'].squeeze(),
            'query_attention_mask': query_encoded['attention_mask'].squeeze(),
            'doc_input_ids': doc_encoded['input_ids'].squeeze(),
            'doc_attention_mask': doc_encoded['attention_mask'].squeeze(),
            'combined_input_ids': combined_encoded['input_ids'].squeeze(),
            'combined_attention_mask': combined_encoded['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class KnowledgeDistillationTrainer:
    """
    Trainer for knowledge distillation from teacher to student bi-encoder
    """
    
    def __init__(self, teacher_model: TeacherModel, student_model: BiEncoder,
                 temperature: float = 3.0, alpha: float = 0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
    def compute_distillation_loss(self, student_scores: torch.Tensor, 
                                teacher_scores: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        # Soften predictions using temperature
        student_soft = F.log_softmax(student_scores / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_scores / self.temperature, dim=1)
        
        # KL divergence loss
        distillation_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distillation_loss *= (self.temperature ** 2)
        
        return distillation_loss
    
    def compute_task_loss(self, student_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute task-specific loss (contrastive learning)"""
        # Create positive and negative pairs
        batch_size = student_scores.size(0)
        
        # Diagonal elements are positive pairs
        positive_scores = torch.diag(student_scores)
        
        # Off-diagonal elements are negative pairs
        mask = torch.eye(batch_size, device=student_scores.device).bool()
        negative_scores = student_scores.masked_select(~mask).view(batch_size, -1)
        
        # Contrastive loss
        positive_loss = -torch.log(torch.sigmoid(positive_scores)).mean()
        negative_loss = -torch.log(1 - torch.sigmoid(negative_scores)).mean()
        
        return positive_loss + negative_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Single training step"""
        self.student_model.train()
        
        # Forward pass through student
        student_outputs = self.student_model(
            query_input_ids=batch['query_input_ids'],
            query_attention_mask=batch['query_attention_mask'],
            doc_input_ids=batch['doc_input_ids'],
            doc_attention_mask=batch['doc_attention_mask']
        )
        
        # Forward pass through teacher
        with torch.no_grad():
            teacher_scores = self.teacher_model(
                input_ids=batch['combined_input_ids'],
                attention_mask=batch['combined_attention_mask']
            )
        
        # Compute losses
        student_scores = student_outputs['scores']
        
        # Distillation loss
        distillation_loss = self.compute_distillation_loss(student_scores, teacher_scores.unsqueeze(1))
        
        # Task loss
        task_loss = self.compute_task_loss(student_scores, batch['label'])
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'distillation_loss': distillation_loss.item(),
            'task_loss': task_loss.item()
        }
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        total_losses = {'total_loss': 0.0, 'distillation_loss': 0.0, 'task_loss': 0.0}
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_step(batch, optimizer)
            
            for key, value in losses.items():
                total_losses[key] += value
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{num_batches}: "
                          f"Total Loss: {losses['total_loss']:.4f}, "
                          f"Distillation Loss: {losses['distillation_loss']:.4f}, "
                          f"Task Loss: {losses['task_loss']:.4f}")
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
            
        return total_losses

# Example usage and training setup
def create_sample_data(num_samples: int = 1000) -> Tuple[List[str], List[str], List[float]]:
    """Create sample training data"""
    queries = [f"Sample query {i}" for i in range(num_samples)]
    documents = [f"Sample document {i} with relevant content" for i in range(num_samples)]
    labels = np.random.rand(num_samples).tolist()  # Random relevance scores
    
    return queries, documents, labels

def main():
    """Main training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use BGE tokenizer - compatible with BERT tokenization
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    
    # Create models with SOTA BGE architectures
    teacher_model = TeacherModel('BAAI/bge-large-en-v1.5').to(device)
    student_model = BiEncoder('BAAI/bge-m3').to(device)
    
    # Create sample data
    queries, documents, labels = create_sample_data(1000)
    
    # Create dataset and dataloader
    dataset = QueryDocDataset(queries, documents, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Reduced batch size for larger models
    
    # Create trainer
    trainer = KnowledgeDistillationTrainer(teacher_model, student_model, temperature=3.0, alpha=0.7)
    
    # Setup optimizer with lower learning rate for pretrained BGE models
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_losses = trainer.train_epoch(dataloader, optimizer)
        
        logger.info(f"Epoch {epoch + 1} completed. Average losses: {epoch_losses}")
    
    # Save student model
    torch.save(student_model.state_dict(), 'distilled_bge_bi_encoder.pth')
    logger.info("Training completed. Model saved as 'distilled_bge_bi_encoder.pth'")

if __name__ == "__main__":
    main()