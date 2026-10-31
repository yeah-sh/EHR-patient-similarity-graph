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

``TemporalGraphAE`` (Phase 7b) is the same encoder trained without labels::

    Z, states   = TemporalGCN.sequence(X, A_hat)  # states[:, t]: GRU state after window t
    link logits = z_i . z_j on Z_t                # window t's edges vs sampled non-edges
    next logits = linear(states[:, t])            # codes active in window t + 1
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

from ehrgraph.config import GraphAEConfig, ModelConfig
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
        Z, states = self.sequence(features, adjacency)
        return Z, states[:, -1]

    def sequence(
        self, features: Sequence[Tensor], adjacency: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Embed each window and return the GRU hidden state after every window.

        The state after window t reads windows 1..t only, never a later one.

        Args:
            features: Dense [N, in_dim] per window, oldest first.
            adjacency: Sparse [N, N] ``A_hat`` per window, same order.

        Returns:
            Z, [N, T, embed_dim], and the GRU states, [N, T, embed_dim]; the last
            state is the trajectory embedding h.

        Raises:
            ValueError: If there are no windows or the two sequences differ in length.
        """
        if not features or len(features) != len(adjacency):
            raise ValueError(
                f"need one adjacency per window, got {len(features)} feature matrices and "
                f"{len(adjacency)} adjacencies"
            )
        Z = torch.stack([self.encoder(X, A) for X, A in zip(features, adjacency)], dim=1)
        states, _ = self.gru(Z)  # one layer: states[:, -1] is the final hidden state
        return Z, states


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


@dataclass(frozen=True)
class GraphAEOutput:
    """Output of ``TemporalGraphAE`` for N nodes, T windows and C codes.

    Attributes:
        window_embeddings: Z, [N, T, embed_dim]: the spatial encoder's output per
            window, decoded into that window's edges.
        states: The GRU state after each window, [N, T, embed_dim].
        next_logits: [N, T - 1, C]: from the state after window t, the logits of
            each code being active in window t + 1.
    """

    window_embeddings: Tensor
    states: Tensor
    next_logits: Tensor

    @property
    def trajectory_embeddings(self) -> Tensor:
        """h, [N, embed_dim]: the GRU state after the last window."""
        return self.states[:, -1]


class GraphAELoss(NamedTuple):
    """``total = link + next_window_weight * next_window`` (see ``graph_ae_loss``)."""

    total: Tensor
    link: Tensor
    next_window: Tensor


class TemporalGraphAE(nn.Module):
    """Self-supervised ``TemporalGCN``: per-window link reconstruction and next-window codes.

    The encoder is a ``TemporalGCN`` (a shared ``SpatialEncoder`` per window, then a
    GRU), with no demographics and no onset head. Two decoders:

    * links: ``sigmoid(z_i . z_j)`` on window t's Z_t (``link_loss``);
    * next window: a linear map from the GRU state after window t to the logits
      of the codes active in window t + 1.

    With ``detach_next_window`` the next-window decoder reads the states with their
    gradient cut: it is then a linear readout trained alongside (for early stopping),
    and only the link term shapes the encoder and the GRU (which then stays as
    initialized).
    """

    def __init__(
        self,
        in_dim: int,
        n_codes: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float,
        detach_next_window: bool = False,
    ) -> None:
        """Create the model.

        Args:
            in_dim: Number of input features per node.
            n_codes: Codes of the next-window target (the feature code block).
            hidden_dim: Width of the hidden GCN layer.
            embed_dim: Width of Z_t and of the GRU states.
            dropout: Dropout rate between the GCN layers.
            detach_next_window: Cut the next-window gradient from the states.
        """
        super().__init__()
        self.detach_next_window = detach_next_window
        self.temporal = TemporalGCN(in_dim, hidden_dim, embed_dim, dropout)
        self.next_window = nn.Linear(embed_dim, n_codes)

    def forward(self, features: Sequence[Tensor], adjacency: Sequence[Tensor]) -> GraphAEOutput:
        """Embed every node in every window and decode the next window's codes.

        Args:
            features: Dense [N, in_dim] per window, oldest first.
            adjacency: Sparse [N, N] ``A_hat`` per window, same order.

        Returns:
            The embeddings, GRU states and next-window logits of all N nodes.
        """
        Z, states = self.temporal.sequence(features, adjacency)
        source = states[:, :-1]
        if self.detach_next_window:
            source = source.detach()
        return GraphAEOutput(Z, states, self.next_window(source))


def build_graph_ae(
    ae_cfg: GraphAEConfig, in_dim: int, n_codes: int, link_only: bool = False
) -> TemporalGraphAE:
    """Create a ``TemporalGraphAE`` with the configured widths and dropout.

    Seed the RNG first (``utils.set_seed``) for reproducible initial weights.

    Args:
        ae_cfg: The ``graph_ae`` section of the configuration.
        in_dim: Number of input features per node.
        n_codes: Codes of the next-window target.
        link_only: Detach the next-window decoder (``TemporalGraphAE``).

    Returns:
        The untrained model.
    """
    return TemporalGraphAE(in_dim, n_codes, ae_cfg.hidden_dim, ae_cfg.embed_dim, ae_cfg.dropout,
                           detach_next_window=link_only)


class LinkSampler:
    """Positive edges and uniformly sampled non-edges of each window, for ``link_loss``.

    Positives are each window's undirected edges, once each (upper triangle of
    ``W``). Each ``sample`` draws as many negatives per window: pairs of distinct
    nodes, both with at least one edge in that window (isolated nodes are skipped),
    drawn uniformly and redrawn while they are an edge. Pairs are stored with the
    smaller node first. A window without edges gives no pairs.
    """

    def __init__(self, weights: Sequence[sp.spmatrix], seed: int) -> None:
        """Index each window's edges and connected nodes.

        Args:
            weights: Symmetric (N, N) ``W`` per window, without self-loops.
            seed: Seed of the generator that draws the negatives.

        Raises:
            ValueError: If a window with edges has no non-edge between connected nodes.
        """
        self.rng = np.random.default_rng(seed)
        self._windows = []
        for W in weights:
            W = sp.csr_matrix(W)
            n = W.shape[0]
            upper = sp.triu(W, k=1).tocoo()
            rows, cols = upper.row.astype(np.int64), upper.col.astype(np.int64)
            nodes = np.flatnonzero(np.diff(W.indptr) > 0)
            if len(rows) and len(rows) >= len(nodes) * (len(nodes) - 1) // 2:
                raise ValueError("a window's connected nodes are all linked: no non-edge to sample")
            self._windows.append((torch.from_numpy(np.vstack([rows, cols])), nodes,
                                  np.sort(rows * n + cols), n))

    def sample(self) -> list[tuple[Tensor, Tensor]]:
        """Return (positives, negatives) per window, each a [2, m] index tensor."""
        return [(positives, self._negatives(nodes, keys, n, positives.shape[1]))
                for positives, nodes, keys, n in self._windows]

    def _negatives(self, nodes: np.ndarray, keys: np.ndarray, n: int, m: int) -> Tensor:
        """Draw ``m`` non-edges between distinct ``nodes``; ``keys`` encode the edges."""
        found = np.empty((2, 0), dtype=np.int64)
        while found.shape[1] < m:
            draw = self.rng.choice(nodes, size=(2, m - found.shape[1]))
            low, high = draw.min(axis=0), draw.max(axis=0)
            keep = (low != high) & ~np.isin(low * n + high, keys)
            found = np.hstack([found, np.vstack([low, high])[:, keep]])
        return torch.from_numpy(found)


def link_logits(Z: Tensor, pairs: Tensor) -> Tensor:
    """Inner-product decoder: ``z_i . z_j`` for each column (i, j) of ``pairs`` [2, m]."""
    return (Z[pairs[0]] * Z[pairs[1]]).sum(dim=-1)


def link_loss(window_embeddings: Tensor, pairs: Sequence[tuple[Tensor, Tensor]]) -> Tensor:
    """Mean over windows with edges of the link-reconstruction BCE.

    In window t, the BCE of ``sigmoid(z_i . z_j)`` (``link_logits`` of Z_t) against 1
    for the positives and 0 for the negatives, averaged over both. Windows without
    edges are skipped; with none at all the loss is 0.

    Args:
        window_embeddings: Z, [N, T, d].
        pairs: (positives, negatives) per window (``LinkSampler.sample``).

    Returns:
        A scalar tensor.
    """
    losses = []
    for t, (positives, negatives) in enumerate(pairs):
        if positives.shape[1] == 0:
            continue
        Z = window_embeddings[:, t]
        logits = link_logits(Z, torch.cat([positives, negatives], dim=1))
        targets = torch.cat([torch.ones(positives.shape[1]), torch.zeros(negatives.shape[1])])
        losses.append(F.binary_cross_entropy_with_logits(logits, targets))
    if not losses:
        return window_embeddings.new_zeros(())
    return torch.stack(losses).mean()


def graph_ae_loss(
    output: GraphAEOutput,
    pairs: Sequence[tuple[Tensor, Tensor]],
    next_targets: Tensor,
    pos_weight: Tensor,
    next_window_weight: float,
) -> GraphAELoss:
    """Return ``link_loss + next_window_weight * masked BCE(next window)``.

    Restrict the next-window term to some patients (a split) by setting the other
    patients' targets to NaN: they still pass messages and enter the link term.

    Args:
        output: ``TemporalGraphAE`` output.
        pairs: (positives, negatives) per window.
        next_targets: [N, T - 1, C]: 1.0, 0.0 or NaN (masked).
        pos_weight: [C] ``positive_weights`` of the training targets.
        next_window_weight: Weight of the next-window term.

    Returns:
        The total and its two terms.
    """
    link = link_loss(output.window_embeddings, pairs)
    next_window = masked_bce(output.next_logits, next_targets, pos_weight)
    return GraphAELoss(link + next_window_weight * next_window, link, next_window)


def pc1_share(X: Tensor | np.ndarray) -> float:
    """Share of the variance of the rows of ``X`` [n, d] on their first principal component.

    1 means every row lies on one line (a collapsed embedding); 1 / d is the floor.
    NaN if all rows are equal.
    """
    X = np.asarray(X.detach().numpy() if isinstance(X, Tensor) else X, dtype=np.float64)
    singular = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)  # n x d, not n x n
    total = np.sum(singular**2)
    return float(singular[0] ** 2 / total) if total > 0 else float("nan")


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
