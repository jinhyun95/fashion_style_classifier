import torch
import torch.nn as nn


class RatioLoss(nn.Module):
    def __init__(self):
        super(RatioLoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')
        self.celoss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, baselines, labels):
        if len(labels.size()) == 2:
            ours = self.bceloss(logits, labels).sum(dim=1)
            base = self.bceloss(baselines, labels).sum(dim=1)
        else:
            ours = self.celoss(logits, labels).sum(dim=1)
            base = self.celoss(baselines, labels).sum(dim=1)
        return (ours / base).mean()


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        bce = self.bceloss(logits, labels)
        modulator = torch.where(labels == 1., - torch.sigmoid(logits) + 1., torch.sigmoid(logits)).pow(self.gamma)
        return (bce * modulator).mean()


class AnchorBCELoss(nn.Module):
    def __init__(self, gamma=2):
        super(AnchorBCELoss, self).__init__()
        self.gamma = gamma
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        bce = self.bceloss(logits, labels)
        pos_p = torch.sigmoid(logits)
        neg_p = torch.sigmoid(logits) * -1. + 1.
        pos_q = (pos_p * (labels == 0.).to(torch.float32)).max(dim=1, keepdim=True)[0]
        neg_q = (neg_p * (labels == 1.).to(torch.float32)).max(dim=1, keepdim=True)[0]
        modulator = torch.where(labels == 1., -pos_p + pos_q + 1., -neg_p + neg_q + 1.).pow(self.gamma)
        return (bce * modulator).mean()


class AnchorCELoss(nn.Module):
    def __init__(self, gamma=2):
        super(AnchorCELoss, self).__init__()
        self.gamma = gamma
        self.celoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        ce = self.celoss(logits, labels).sum(1)
        prob = torch.softmax(logits, 1)
        top2 = torch.topk(prob, dim=1, k=2)
        # q_star = max(prob) if indice(max(prob)) != truth else top2(prob)
        # modulator = (1 + q - q_star) ** gamma
        q_star = torch.where(top2[1][0] == labels, top2[0][1], top2[0][0])
        modulator = (torch.gather(prob, 1, labels.unsqueeze(1)).squeeze(1) - q_star + 1.).pow(self.gamma)
        return (ce * modulator).mean()
