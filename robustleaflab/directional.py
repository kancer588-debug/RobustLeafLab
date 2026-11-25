"""Apply the development-locked early rescue and directional subtype votes."""

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import verify


HEADS_DIR = Path(__file__).with_name("directional_weights")
EMBED_DIM = 384
HIDDEN_DIM = 48


class EarlyRescueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.region = nn.Sequential(nn.Linear(EMBED_DIM, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1))
        self.context = nn.Linear(EMBED_DIM, 1)

    def forward(self, whole, regions):
        evidence = self.region(regions).squeeze(-1)
        temperature = 0.35
        pooled = temperature * (torch.logsumexp(evidence / temperature, dim=1) - math.log(evidence.shape[1]))
        return pooled + self.context(whole).flatten()


class DirectionalSubtypeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.LayerNorm(2 * EMBED_DIM), nn.Linear(2 * EMBED_DIM, HIDDEN_DIM), nn.GELU())
        self.output = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, features, late_leads):
        return self.output(self.image(features)).gather(1, late_leads.long().unsqueeze(1)).flatten()


class DirectionalReview(nn.Module):
    def __init__(self):
        super().__init__()
        self.settings = json.loads((HEADS_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.early_rescue = nn.ModuleList()
        self.subtype = nn.ModuleList()
        for setting in self.settings["heads"]:
            early = setting["early_rescue"]
            rescue = EarlyRescueHead()
            rescue.load_state_dict(self.load(early)["head"], strict=True)
            self.early_rescue.append(rescue)
            subtype = DirectionalSubtypeHead()
            subtype.load_state_dict(self.load(setting["subtype"]), strict=True)
            self.subtype.append(subtype)

    @staticmethod
    def load(setting):
        path = verify(HEADS_DIR / setting["file"], setting["sha256"])
        return torch.load(path, map_location="cpu", weights_only=True)

    def forward(self, scores, whole, patches, features):
        side = math.isqrt(patches.shape[1])
        grid = patches.transpose(1, 2).reshape(len(patches), EMBED_DIM, side, side)
        regions = F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2)
        healthy = scores.argmax(1) == 0
        health_gap = scores[:, 0] - scores[:, 1:].amax(1)
        subtype_gap = (scores[:, 1] - scores[:, 2]).abs()
        late_leads = scores[:, 2] > scores[:, 1]
        rescue_votes, subtype_votes = [], []
        for card, rescue, subtype in zip(self.settings["heads"], self.early_rescue, self.subtype, strict=True):
            early = card["early_rescue"]
            rescue_votes.append(healthy & (health_gap <= early["gap"]) & (rescue(whole, regions) >= early["threshold"]))
            correction = card["subtype"]
            logit = subtype(features, late_leads)
            enough_evidence = torch.where(late_leads, logit >= correction["early_threshold"], logit <= -correction["late_threshold"])
            subtype_votes.append(~healthy & (subtype_gap <= correction["gap"]) & enough_evidence)
        rescued = torch.stack(rescue_votes).sum(0) >= 2
        switched = torch.stack(subtype_votes).sum(0) >= 2
        updated = scores.clone()
        updated[rescued, 1] = scores[rescued, 0] + 1.
        other_subtype = 3 - scores[switched].argmax(1)
        updated[switched, other_subtype] = scores[switched].amax(1) + 1.
        return updated
