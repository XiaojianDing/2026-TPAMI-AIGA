import torch
import torch.nn.functional as F
import sys

def mi_loss_infoNCE_crossR(R_list, temperature=0.5):
    """
        temperature:
        NGs:temperature=1;
        Fashion:temperature=0.5;
        MNIST-USPS:temperature=0.6;
        Caltech-5V:temperature=0.5;
        Synthetic3d:temperature=0.2;
        Cifar10:temperature=0.2;
    """
    m = len(R_list)
    N = R_list[0].size(0)
    device = R_list[0].device

    total_loss = 0.0
    count = 0

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            Ri = F.normalize(R_list[i], dim=-1)
            Rj = F.normalize(R_list[j], dim=-1)

            sim_matrix = torch.matmul(Ri, Rj.t()) / temperature


            labels = torch.arange(N, device=device)

            loss_ij = F.cross_entropy(sim_matrix, labels)

            total_loss += loss_ij
            count += 1

    return total_loss / count






