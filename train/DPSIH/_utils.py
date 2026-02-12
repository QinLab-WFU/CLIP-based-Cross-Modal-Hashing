import torch


def mean_average_precision(qB, rB, qL, rL, topk=None, rank=None):
    num_query = qL.shape[0]
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    if topk is None:
        topk = rL.shape[0]
    mean_AP = 0.0
    for i in range(num_query):
        retrieval = (qL[i, :] @ rL.T > 0).float()
        _, K, D = qB.shape
        sim_kk = qB[i] @ rB.view(-1, D).T
        sim_kk = sim_kk.view(1, K, rB.size(0), K)
        sim_kk = sim_kk.permute(0, 1, 3, 2).contiguous()
        sim_kk = sim_kk.view(1, -1, rB.size(0))
        sim, _ = sim_kk.max(dim=1)
        sim = sim.flatten()
        hamming_dist = 0.5 * (D - sim)
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]
        retrieval_cnt = retrieval.sum().int().item()
        if retrieval_cnt == 0:
            continue
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(retrieval.device)
        index = ((retrieval == 1).nonzero(as_tuple=False).squeeze() + 1.0).float()
        mean_AP += (score / index).mean()
    mean_AP = mean_AP / num_query
    return mean_AP