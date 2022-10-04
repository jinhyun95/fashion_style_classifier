import torch


class TopKAnyAccuracy:
    def __init__(self, k):
        self.k = k
    def __call__(self, logits, labels):
        topk_indices = torch.topk(logits, k=self.k, dim=1, largest=True)[1]
        if len(labels.size()) == 1:
            # single label accuracy
            accuracy = (topk_indices == logits.unsqueeze(1).expand_as(topk_indices)).sum() / float(labels.size(0))
        elif len(labels.size()) == 2:
            # multi label accuracy
            accuracy = torch.gather(labels, 1, topk_indices).sum(1).clamp(max=1.).sum() / float(labels.size(0))
        else:
            raise NotImplementedError
        return accuracy.item()


class TopKMainOnlyAccuracy:
    def __init__(self, k):
        self.k = k
    def __call__(self, logits, labels):
        assert logits.size(0) == labels.size(0)
        assert len(labels.size()) == 1
        topk_indices = torch.topk(logits, k=self.k, dim=1, largest=True)[1]
        accuracy = (topk_indices == labels.unsqueeze(1).expand_as(topk_indices)).sum() / float(labels.size(0))
        return accuracy.item()


class TopKAllAccuracy:
    def __init__(self, k):
        self.k = k
    def __call__(self, logits, labels):
        assert logits.size(0) == labels.size(0)
        topk_indices = torch.topk(logits, k=self.k, dim=1, largest=True)[1]
        k_expanded = torch.ones_like(labels[:, 0]) * self.k
        num_answers = torch.where(labels.sum(1) > k_expanded, k_expanded, labels.sum(1))
        accuracy = (torch.gather(labels, 1, topk_indices).sum(1) == num_answers).sum() / float(labels.size(0))
        return accuracy.item()
