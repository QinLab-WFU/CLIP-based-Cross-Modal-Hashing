import torch
import torch.nn as nn


def get_matches_and_diffs(labels):
    matches = (labels.float() @ labels.float().T).byte()
    diffs = matches ^ 1  # 异或运算得到负标签的矩阵
    return matches, diffs


def get_all_triplets_indices_vectorized_method(all_matches, all_diffs):
    """
    Args:
        all_matches (torch.Tensor): 相同标签
        all_diffs (torch.Tensor): 不相同标签

    Processing : all_matches.unsqueeze(2) -> [Batch,Batch,1]
                 all_diffs.unsqeeeze(1) -> [Batch,1,Batch]

    Returns:
        torch.Tensor: _description_
    """

    triplets = all_matches.unsqueeze(2) * all_diffs.unsqueeze(1)
    return torch.where(triplets)


class TripletMinner(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sim_mat = get_matches_and_diffs
        self.selctor = get_all_triplets_indices_vectorized_method

    def forward(self, labels):
        a, b = self.sim_mat(labels)
        c = self.selctor(a, b)

        return c
