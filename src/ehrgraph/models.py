"""Temporal graph convolutional network over the per-window patient-similarity graphs.

Pure PyTorch, no PyTorch Geometric. For the training-period windows t = 1..T
(2017-2022 for the project cutoff)::

    Z_t = SpatialEncoder(X_t, A_hat_t)     # [N, embed_dim]; one encoder for every t
    Z   = stack(Z_1, ..., Z_T)             # [N, T, embed_dim]: the time-series embedding
    h   = final GRU hidden state over Z    # [N, embed_dim]: the trajectory embedding
    u   = [h | age, sex]                   # with model.use_demographics, else u = h
    any_onset logit    = primary head(u)   # [N]
    onset group logits = secondary head(u) # [N, G]

``A_hat_t = D^(-1/2) (W_t + I) D^(-1/2)`` is the Phase 4 adjacency as a torch
sparse tensor (``sparse_tensor``); an isolated node keeps only its self-loop. Every
node takes part in message passing. Which labels enter a loss is set by the
targets alone: a NaN target is masked and contributes nothing (``masked_bce``).
Demographics enter the heads only: never the graphs or the node features, so the
embeddings Z and h do not depend on them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ehrgraph.config import ModelConfig
from ehrgraph.features import DEMOGRAPHICS
from ehrgraph.graph import laplacian


@dataclass(frozen=True)
class ModelOutput:
    """Output of ``OnsetModel`` for N nodes, T windows and G onset groups.

    Attributes:
        window_embeddings: Z, [N, T, embed_dim]: one embedding per patient and window.
        trajectory_embeddings: h, [N, embed_dim]: the final GRU hidden state.
        any_onset_logits: [N], logit of the primary target ``any_onset``.
        onset_logits: [N, G], logits of the per-group onset targets.
    """

    window_embeddings: Tensor
    trajectory_embeddings: Tensor
    any_onset_logits: Tensor
    onset_logits: Tensor


class LossParts(NamedTuple):
    """``total = primary + secondary_weight * secondary`` (see ``onset_loss``)."""

    total: Tensor
    primary: Tensor
    secondary: Tensor


class GCNLayer(nn.Module):
    """Graph convolution ``H' = A_hat @ (H @ W) + b`` with a sparse ``A_hat``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Create the layer with Xavier-uniform weights and a zero bias.

        Args:
            in_dim: Width of the input rows H.
            out_dim: Width of the output rows H'.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform initialization of W; zero bias."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, H: Tensor, A_hat: Tensor) -> Tensor:
        """Return ``A_hat @ (H @ W) + b``.

        Args:
            H: Dense node rows, [N, in_dim].
            A_hat: Sparse normalized adjacency, [N, N].

        Returns:
            Dense [N, out_dim].
        """
        return torch.sparse.mm(A_hat, H @ self.weight) + self.bias


class SpatialEncoder(nn.Module):
    """Two GCN layers mapping one window's features to node embeddings.

    ``Z = GCN_2(dropout(relu(GCN_1(X, A_hat))), A_hat)``. ``TemporalGCN`` applies the
    same encoder (the same weights) to every window.
    """

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float) -> None:
        """Create the encoder.

        Args:
            in_dim: Number of input features per node.
            hidden_dim: Width of the hidden layer.
            embed_dim: Width of the output embedding.
            dropout: Dropout rate between the two layers.
        """
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, A_hat: Tensor) -> Tensor:
        """Embed every node of one window.

        Args:
            X: Dense features, [N, in_dim].
            A_hat: Sparse normalized adjacency, [N, N].

        Returns:
            [N, embed_dim].
        """
        return self.conv2(self.dropout(torch.relu(self.conv1(X, A_hat))), A_hat)


class TemporalGCN(nn.Module):
    """A shared ``SpatialEncoder`` per window, then a GRU over the windows."""

    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float) -> None:
        """Create the encoder and a one-layer GRU with ``embed_dim`` units.

        Args:
            in_dim: Number of input features per node.
            hidden_dim: Width of the hidden GCN layer.
            embed_dim: Width of the window and trajectory embeddings.
            dropout: Dropout rate between the GCN layers.
        """
        super().__init__()
        self.encoder = SpatialEncoder(in_dim, hidden_dim, embed_dim, dropout)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def forward(
        self, features: Sequence[Tensor], adjacency: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Embed each window, then run the GRU over windows 1..T.

        Args:
            features: Dense [N, in_dim] per window, oldest first.
            adjacency: Sparse [N, N] ``A_hat`` per window, same order.

        Returns:
            Z, [N, T, embed_dim], and h, the final hidden state, [N, embed_dim].

        Raises:
            ValueError: If there are no windows or the two sequences differ in length.
        """
        if not features or len(features) != len(adjacency):
            raise ValueError(
                f"need one adjacency per window, got {len(features)} feature matrices and "
                f"{len(adjacency)} adjacencies"
            )
        Z = torch.stack([self.encoder(X, A) for X, A in zip(features, adjacency)], dim=1)
        _, h_n = self.gru(Z)
        return Z, h_n[-1]


class OnsetHeads(nn.Module):
    """Linear heads on the head input: ``any_onset`` and the onset groups."""

    def __init__(self, in_dim: int, n_groups: int) -> None:
        """Create the primary (1 logit) and secondary (``n_groups`` logits) heads.

        Args:
            in_dim: Width of the head input: the trajectory embedding, plus the
                demographics when they are used.
            n_groups: Number of onset groups.
        """
        super().__init__()
        self.primary = nn.Linear(in_dim, 1)
        self.secondary = nn.Linear(in_dim, n_groups)

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """Return the ``any_onset`` logits [N] and the group logits [N, G]."""
        return self.primary(u).squeeze(-1), self.secondary(u)


class OnsetModel(nn.Module):
    """``TemporalGCN`` plus ``OnsetHeads``: the Phase 6 model. No mortality head."""

    def __init__(
        self,
        in_dim: int,
        n_groups: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float,
        n_demographics: int = 0,
    ) -> None:
        """Create the model.

        Args:
            in_dim: Number of input features per node.
            n_groups: Number of onset groups (secondary targets).
            hidden_dim: Width of the hidden GCN layer.
            embed_dim: Width of the window and trajectory embeddings.
            dropout: Dropout rate between the GCN layers.
            n_demographics: Patient-level inputs concatenated to the trajectory
                embedding before the heads; 0 for none.
        """
        super().__init__()
        self.n_demographics = n_demographics
        self.temporal = TemporalGCN(in_dim, hidden_dim, embed_dim, dropout)
        self.heads = OnsetHeads(embed_dim + n_demographics, n_groups)

    def forward(
        self,
        features: Sequence[Tensor],
        adjacency: Sequence[Tensor],
        demographics: Tensor | None = None,
    ) -> ModelOutput:
        """Embed every node in every window and score the onset targets.

        Args:
            features: Dense [N, in_dim] per window, oldest first.
            adjacency: Sparse [N, N] ``A_hat`` per window, same order.
            demographics: [N, n_demographics], required when ``n_demographics`` > 0
                and ignored otherwise.

        Returns:
            The embeddings and logits of all N nodes.

        Raises:
            ValueError: If the model uses demographics and they are missing or of
                the wrong shape.
        """
        Z, h = self.temporal(features, adjacency)
        u = h
        if self.n_demographics:
            if demographics is None or demographics.shape != (h.shape[0], self.n_demographics):
                shape = None if demographics is None else tuple(demographics.shape)
                raise ValueError(
                    f"expected demographics of shape ({h.shape[0]}, {self.n_demographics}), "
                    f"got {shape}"
                )
            u = torch.cat([h, demographics], dim=1)
        any_onset, onsets = self.heads(u)
        return ModelOutput(Z, h, any_onset, onsets)


def build_model(model_cfg: ModelConfig, in_dim: int, n_groups: int) -> OnsetModel:
    """Create an ``OnsetModel`` with the configured widths, dropout and head inputs.

    Seed the RNG first (``utils.set_seed``) for reproducible initial weights. With
    ``model.use_demographics`` the heads also take ``features.DEMOGRAPHICS``.

    Args:
        model_cfg: The ``model`` section of the configuration.
        in_dim: Number of input features per node.
        n_groups: Number of onset groups.

    Returns:
        The untrained model.
    """
    return OnsetModel(
        in_dim, n_groups, model_cfg.hidden_dim, model_cfg.embed_dim, model_cfg.dropout,
        len(DEMOGRAPHICS) if model_cfg.use_demographics else 0,
    )


class EdgeDropper:
    """DropEdge: fresh adjacencies with a random fraction of each window's edges removed.

    Each ``sample`` drops every undirected edge of every ``W_t`` independently with
    probability ``rate`` (both directions together, so the graph stays symmetric)
    and renormalizes ``A_hat = D^(-1/2) (W' + I) D^(-1/2)``, D the degrees of
    ``W' + I``, as ``graph.normalize_adjacency`` does. For training only; evaluate
    on the full graphs.
    """

    def __init__(self, weights: Sequence[sp.spmatrix], rate: float, seed: int) -> None:
        """Store the upper-triangle edges of each window.

        Args:
            weights: Symmetric non-negative (N, N) ``W`` per window, without self-loops.
            rate: Fraction of edges to drop, in [0, 1).
            seed: Seed of the generator that draws the edges to drop.

        Raises:
            ValueError: If ``rate`` is outside [0, 1).
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"the DropEdge rate must lie in [0, 1), got {rate}")
        self.rate = rate
        self.generator = torch.Generator().manual_seed(seed)
        self._edges = []
        for W in weights:
            upper = sp.triu(W, k=1).tocoo()
            self._edges.append((
                torch.from_numpy(upper.row.astype(np.int64)),
                torch.from_numpy(upper.col.astype(np.int64)),
                torch.from_numpy(upper.data.astype(np.float32)),
                W.shape[0],
            ))

    def sample(self) -> list[Tensor]:
        """Return one renormalized sparse ``A_hat`` per window, edges dropped at random."""
        adjacency = []
        for row, col, weight, n in self._edges:
            keep = torch.rand(len(weight), generator=self.generator) >= self.rate
            row, col, weight = row[keep], col[keep], weight[keep]
            loops = torch.arange(n)
            rows = torch.cat([row, col, loops])
            cols = torch.cat([col, row, loops])
            values = torch.cat([weight, weight, torch.ones(n)])
            degree = torch.zeros(n).index_add_(0, rows, values)  # >= 1: the self-loop
            inverse_sqrt = degree.rsqrt()
            values = values * inverse_sqrt[rows] * inverse_sqrt[cols]
            adjacency.append(
                torch.sparse_coo_tensor(
                    torch.stack([rows, cols]), values, (n, n), check_invariants=True
                ).coalesce()
            )
        return adjacency


def sparse_tensor(matrix: sp.spmatrix) -> Tensor:
    """Convert a scipy sparse matrix to a coalesced float32 torch sparse COO tensor."""
    coo = sp.coo_matrix(matrix)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, check_invariants=True).coalesce()


def masked_bce(logits: Tensor, targets: Tensor, pos_weight: Tensor) -> Tensor:
    """Binary cross-entropy with logits over the observed (non-NaN) targets only.

    Masked (NaN) entries contribute neither to the loss nor to its gradient. The
    loss is the mean over observed entries (over every column together for 2-D
    targets); it is 0 if nothing is observed.

    Args:
        logits: [N] or [N, G].
        targets: Same shape: 1.0, 0.0 or NaN (masked).
        pos_weight: Weight of the positive term: a scalar, or [G] for [N, G] targets.

    Returns:
        A scalar tensor.
    """
    seen = ~torch.isnan(targets)
    losses = F.binary_cross_entropy_with_logits(
        logits, torch.where(seen, targets, 0.0), pos_weight=pos_weight, reduction="none"
    )
    return torch.where(seen, losses, 0.0).sum() / seen.sum().clamp(min=1)


def positive_weights(targets: Tensor) -> Tensor:
    """Return ``negatives / positives`` per column of the observed targets.

    Use on training targets only. A column with no positive or no negative gets 1.

    Args:
        targets: [N] or [N, G]: 1.0, 0.0 or NaN (masked).

    Returns:
        A scalar for [N] targets, else [G].
    """
    positives = (targets == 1).sum(dim=0).to(torch.float32)
    negatives = (targets == 0).sum(dim=0).to(torch.float32)
    defined = (positives > 0) & (negatives > 0)
    return torch.where(defined, negatives / positives.clamp(min=1), 1.0)


def onset_loss(
    output: ModelOutput,
    any_onset: Tensor,
    onsets: Tensor,
    any_onset_pos_weight: Tensor,
    onset_pos_weight: Tensor,
    secondary_weight: float,
) -> LossParts:
    """Return ``BCE(any_onset) + secondary_weight * masked BCE(onset groups)``.

    Restrict a loss to some patients (a split) by setting the other patients'
    targets to NaN: masked patients still pass messages, but add nothing here.

    Args:
        output: ``OnsetModel`` output.
        any_onset: [N] targets: 1.0, 0.0 or NaN.
        onsets: [N, G] targets: 1.0, 0.0 or NaN.
        any_onset_pos_weight: Scalar ``positive_weights`` of the training targets.
        onset_pos_weight: [G] ``positive_weights`` of the training targets.
        secondary_weight: Weight of the group term (``model.secondary_weight``).

    Returns:
        The total and its two terms.
    """
    primary = masked_bce(output.any_onset_logits, any_onset, any_onset_pos_weight)
    secondary = masked_bce(output.onset_logits, onsets, onset_pos_weight)
    return LossParts(primary + secondary_weight * secondary, primary, secondary)


def graph_smoothness(
    window_embeddings: Tensor, laplacians: Sequence[Tensor], connected: Sequence[Tensor]
) -> Tensor:
    """Mean over windows of the normalized Dirichlet energy of Z_t on graph t.

    For window t, ``trace(Z_t^T L_t Z_t) / trace(Z_t^T Z_t)``, with both traces over
    the nodes that have an edge: ``L_t`` is ``smoothness_laplacian(W_t)``, whose rows
    of isolated nodes are zero. It is the Rayleigh quotient of L_t, in [0, 2], so the
    model cannot lower it by shrinking Z; it is 0 when each embedding column is
    proportional to sqrt(degree) within each connected component.

    Args:
        window_embeddings: Z, [N, T, d].
        laplacians: Sparse [N, N] per window.
        connected: Bool [N] per window: nodes with at least one edge.

    Returns:
        A scalar tensor.
    """
    energies = []
    for t, (L, has_edge) in enumerate(zip(laplacians, connected)):
        Z = window_embeddings[:, t, :]
        numerator = (Z * torch.sparse.mm(L, Z)).sum()
        denominator = Z[has_edge].pow(2).sum().clamp(min=torch.finfo(Z.dtype).tiny)
        energies.append(numerator / denominator)
    return torch.stack(energies).mean()


def smoothness_laplacian(W: sp.spmatrix) -> sp.csr_matrix:
    """Return ``I_E - D^(-1/2) W D^(-1/2)``, ``I_E`` the identity on nodes with an edge.

    The normalized Laplacian (``graph.laplacian``) with the rows and columns of
    isolated nodes zeroed: with no neighbour to be smooth with, they are left out
    of ``graph_smoothness``.

    Args:
        W: Symmetric non-negative (n, n) adjacency without self-loops.

    Returns:
        CSR (n, n), positive semi-definite.
    """
    isolated = np.diff(sp.csr_matrix(W).indptr) == 0
    return sp.csr_matrix(laplacian(W) - sp.diags(isolated.astype(np.float64)))
