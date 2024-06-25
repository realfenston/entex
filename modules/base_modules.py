import torch
import torch.nn as nn


class XYZConverter(nn.Module):
    def __init__(self):
        super(XYZConverter, self).__init__()
        self.basexyzs = torch.tensor([(-0.5272, 1.3593, 0.000), (0.000, 0.000, 0.000), (1.5233, 0.000, 0.000)])

    def compute_all_atom(self, Rs, Ts):
        B, L = Rs.shape[:2]
        RTF0 = torch.eye(4).repeat(B,L,1,1).to(device=Rs.device)
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts
        basexyzs = self.basexyzs[None, None, ...].repeat(B, L, 1, 1).to(Rs.device)
        xyzs = torch.einsum('brij,brmj->brmi', Rs, basexyzs - Ts[..., None, :].repeat(1,1,3,1))
        return RTF0, xyzs[..., :3]