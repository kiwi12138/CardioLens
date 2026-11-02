"""
Task-Aware MIL model with Transformer Attention for multi-sequence
cardiac MRI slice importance estimation.

Architecture follows Section C.1 of the paper:
    1. Shared ResNet-18 encoder (1-channel adapted, ImageNet pretrained).
    2. Learnable positional encoding to preserve slice ordering.
    3. Per-task attention module:
         - Multi-head self-attention across slices (inter-slice context).
         - Task-query cross-attention (task-conditioned summary).
         - MLP scorer with sigmoid activation (per-slice importance).
    4. Per-task two-layer MLP classifier on the weighted-sum bag feature.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class SharedEncoder(nn.Module):
    """Shared 2D backbone applied independently to each slice."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            # Adapt the first conv to single-channel input.
            resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 512
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1, H, W] -> [N, feature_dim]
        feats = self.features(x).squeeze(-1).squeeze(-1)
        return feats


class PositionalEncoding(nn.Module):
    """
    Positional encoding to preserve spatial / temporal slice ordering.
    Supports both fixed sinusoidal and learnable variants.
    """

    def __init__(self, d_model: int, max_len: int = 1000, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        self.max_len = max_len

        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            n = x.size(0)
            if n > self.max_len:
                raise ValueError(f"Sequence length {n} exceeds max_len {self.max_len}")
            return x + self.pe[0, :n, :]
        n = x.size(1)
        if n > self.max_len:
            raise ValueError(f"Sequence length {n} exceeds max_len {self.max_len}")
        return x + self.pe[:, :n, :]


class TransformerAttentionHead(nn.Module):
    """
    Per-task attention module (paper Section C.1.4).

    Stage 1: self-attention among slices (contextualization).
    Stage 2: task-query cross-attention (task-conditioned summary).
    Stage 3: MLP scorer with sigmoid activation (per-slice weights in (0, 1)).

    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_self_attention: bool = True,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.use_self_attention = use_self_attention

        # Learnable task query (one per task).
        self.task_query = nn.Parameter(torch.randn(1, 1, feature_dim))

        if use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm1 = nn.LayerNorm(feature_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        # Two-layer feed-forward network for the task representation.
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.norm3 = nn.LayerNorm(feature_dim)

        # MLP scorer that maps task-enhanced slice features to scalar logits.
        self.attention_scorer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [N, feature_dim], slice features for one bag.
            mask:     [N], optional key-padding mask (True positions are ignored).

        Returns:
            weights:           [N], per-slice importance in (0, 1).
            attended_features: [N, feature_dim], self-attention output.
            raw_scores:        [N], pre-sigmoid logits.
        """
        # Stage 1: self-attention among slices.
        if self.use_self_attention:
            x = features.unsqueeze(0)  # [1, N, D]
            attn_out, _ = self.self_attn(
                x, x, x,
                key_padding_mask=mask.unsqueeze(0) if mask is not None else None,
            )
            features_sa = self.norm1(x + self.dropout(attn_out)).squeeze(0)  # [N, D]
        else:
            features_sa = features

        # Stage 2: task-query cross-attention.
        task_query = self.task_query                               # [1, 1, D]
        kv = features_sa.unsqueeze(0)                              # [1, N, D]
        cross_out, _ = self.cross_attn(
            task_query, kv, kv,
            key_padding_mask=mask.unsqueeze(0) if mask is not None else None,
        )
        cross_out = self.norm2(task_query + self.dropout(cross_out))
        ffn_out = self.ffn(cross_out)
        task_repr = self.norm3(cross_out + self.dropout(ffn_out))  # [1, 1, D]

        # Stage 3: per-slice scoring on task-enhanced features.
        task_vec = task_repr.squeeze(0).squeeze(0)                 # [D]
        enhanced = features_sa + task_vec.unsqueeze(0)             # [N, D]
        raw_scores = self.attention_scorer(enhanced).squeeze(-1)   # [N]
        weights = torch.sigmoid(raw_scores)                        # [N]

        return weights, features_sa, raw_scores


class TaskSpecificClassifier(nn.Module):
    """Two-layer MLP head producing class logits for one task."""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TaskAwareMILModel(nn.Module):
    """
    Multi-sequence Task-Aware MIL model.

    A shared encoder produces slice features that are routed through per-task
    attention heads and per-task classifiers. The model is trained jointly over
    all sequences (Cine, LGE, T2, Perfusion).
    """

    def __init__(
        self,
        all_task_configs: Dict,
        backbone: str = "resnet18",
        pretrained: bool = True,
        target_k: int = 8,
        num_heads: int = 4,
        use_self_attention: bool = True,
        use_positional_encoding: bool = True,
        learnable_pe: bool = True,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.all_task_configs = all_task_configs
        self.target_k = target_k
        self.use_positional_encoding = use_positional_encoding

        self.encoder = SharedEncoder(backbone, pretrained)
        feature_dim = self.encoder.feature_dim

        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(
                feature_dim, max_len=max_seq_len, learnable=learnable_pe
            )

        self.attention_heads = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for task_name, cfg in all_task_configs.items():
            self.attention_heads[task_name] = TransformerAttentionHead(
                feature_dim,
                num_heads=num_heads,
                use_self_attention=use_self_attention,
            )
            self.classifiers[task_name] = TaskSpecificClassifier(
                feature_dim, cfg["num_classes"]
            )

        print(f"Model created with {len(all_task_configs)} tasks")
        print(f"  Target K: {self.target_k}")
        print(f"  Num heads: {num_heads}")
        print(f"  Use self-attention: {use_self_attention}")
        print(f"  Use positional encoding: {use_positional_encoding}")
        print(
            "  Shared encoder parameters: "
            f"{sum(p.numel() for p in self.encoder.parameters()):,}"
        )

    def forward_sequence(
        self,
        images: torch.Tensor,
        sequence_type: str,
        task_configs: Dict,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict]]:
        """
        Forward pass for one sequence's bag.

        Args:
            images:        [N, 1, H, W], all slices of one (patient, sequence) bag.
            sequence_type: name of the sequence (used by callers; not consumed here).
            task_configs:  task configs whose tasks should be predicted for this bag.

        Returns:
            all_logits:         {task_name -> [num_classes] logits}.
            all_attention_info: {task_name -> {"weights", "scores", "attended_features"}}.
        """
        del sequence_type  # kept in signature for API symmetry with callers

        features = self.encoder(images)
        if self.use_positional_encoding:
            features = self.pos_encoding(features)

        all_logits: Dict[str, torch.Tensor] = {}
        all_attention_info: Dict[str, Dict] = {}

        for task_name in task_configs.keys():
            weights, attended, raw_scores = self.attention_heads[task_name](features)
            bag_feature = torch.sum(weights.unsqueeze(1) * attended, dim=0)  # [D]
            logits = self.classifiers[task_name](bag_feature)
            all_logits[task_name] = logits
            all_attention_info[task_name] = {
                "weights": weights,
                "scores": raw_scores,
                "attended_features": attended,
            }

        return all_logits, all_attention_info

    @torch.no_grad()
    def extract_importance(
        self,
        images: torch.Tensor,
        task_names: list,
    ) -> torch.Tensor:
        """
        Extract task-averaged importance scores for top-K slice selection.
        Implements Algorithm 2 (importance-based slice selection) of the paper.

        Args:
            images:     [N, 1, H, W], slices of one (patient, sequence) bag.
            task_names: tasks associated with this sequence; the returned score
                        is the mean over these tasks' attention weights.

        Returns:
            mean_weights: [N] tensor of per-slice importance scores.
        """
        features = self.encoder(images)
        if self.use_positional_encoding:
            features = self.pos_encoding(features)

        weights_per_task = []
        for task_name in task_names:
            w, _, _ = self.attention_heads[task_name](features)
            weights_per_task.append(w)
        return torch.stack(weights_per_task, dim=0).mean(dim=0)
