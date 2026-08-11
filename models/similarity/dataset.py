"""
models/similarity/dataset.py

Triplet generation for deep metric learning. Anchor/positive share a
position_group; the negative is drawn from a different position_group.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class PlayerTripletDataset(Dataset):
    def __init__(self, features_df, player_positions):
        # features_df: normalized per-90 stats, indexed by player_id
        # player_positions: dict mapping player_id to position_group
        self.features = torch.tensor(features_df.values, dtype=torch.float32)
        self.player_ids = features_df.index.tolist()
        self.positions = player_positions

        self._by_position: dict[str, list[int]] = {}
        for i, pid in enumerate(self.player_ids):
            self._by_position.setdefault(self.positions[pid], []).append(i)

    def __len__(self):
        return len(self.player_ids)

    def __getitem__(self, idx):
        anchor_id = self.player_ids[idx]
        anchor_feat = self.features[idx]
        anchor_pos = self.positions[anchor_id]

        # Positive: random player with the SAME position_group
        pos_candidates = [i for i in self._by_position.get(anchor_pos, []) if i != idx]
        pos_idx = np.random.choice(pos_candidates) if pos_candidates else idx

        # Negative: random player with a DIFFERENT position_group
        neg_candidates = [
            i for pos, idxs in self._by_position.items() if pos != anchor_pos
            for i in idxs
        ]
        neg_idx = np.random.choice(neg_candidates) if neg_candidates else idx

        return anchor_feat, self.features[pos_idx], self.features[neg_idx]
