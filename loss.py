import torch

from .util import get_t, rigid_from_3_points


def compute_general_FAPE(self, X, Y, atom_mask, Z=3.0, dclamp=10.0, eps=1e-4):
    B = X.shape[0]
    """X_x = torch.gather(X, 2, frames[...,0:1].repeat(N,1,1,3))
    X_y = torch.gather(X, 2, frames[...,1:2].repeat(N,1,1,3))
    X_z = torch.gather(X, 2, frames[...,2:3].repeat(N,1,1,3))"""
    X_x = X[..., 0, :]
    X_y = X[..., 1, :]
    X_z = X[..., 2, :]
    uX, tX = rigid_from_3_points(X_x, X_y, X_z)

    """Y_x = torch.gather(Y, 2, frames[...,0:1].repeat(1,1,1,3))
    Y_y = torch.gather(Y, 2, frames[...,1:2].repeat(1,1,1,3))
    Y_z = torch.gather(Y, 2, frames[...,2:3].repeat(1,1,1,3))"""
    Y_x = Y[..., 0, :]
    Y_y = Y[..., 1, :]
    Y_z = Y[..., 2, :]
    uY, tY = rigid_from_3_points(Y_x, Y_y, Y_z)
    """xij = torch.einsum(
        'brji,brsj->brsi',
        uX[:,frame_mask[0]], X[:,atom_mask[0]][:,None,...] - X_y[:,frame_mask[0]][:,:,None,...]
    )
    xij_t = torch.einsum('rji,rsj->rsi', uY[frame_mask], Y[atom_mask][None,...] - Y_y[frame_mask][:,None,...])"""

    xij = torch.einsum('brji,brsj->brsi', uX, X - tX[...,None,:].repeat(1,1,3,1))
    xij_t = torch.einsum('brji,brsj->brsi', uY, Y - tY[...,None,:].repeat(1,1,3,1))
    diff = torch.sqrt(torch.sum(torch.square(xij - xij_t), dim=-1) + eps)
    diff = diff[atom_mask.to(torch.bool())]
    loss = (1.0 / Z) * torch.mean((torch.clamp(diff, max=dclamp)).mean(dim=(1)))
    return loss


def calc_str_loss(pred, true, mask_2d, logit_pae=None, same_chain=None, negative=False, d_clamp=None, d_clamp_inter=30.0, A=10.0, gamma=1.0, eps=1e-6):
    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    pred = pred.unsqueeze(0)
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2], non_ideal=True)
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])
    
    difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
    eij_label = difference[-1].clone().detach()
    
    if d_clamp != None:
        clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
        clamp = clamp[None]
        difference = torch.clamp(difference, max=clamp)
    loss = difference / A # (I, B, L, L)

    # Get a mask information (ignore missing residue + inter-chain residues)
    # for positive cases, mask = mask_2d
    # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
    if same_chain is None:
        mask = mask_2d
    else:
        mask = mask_2d * same_chain
    # calculate masked loss (ignore missing regions when calculate loss)
    #loss = (mask[None]*loss).sum(dim=(1,2,3)) / (mask.sum()+eps) # (I)
    loss = (mask[None,...,None]*loss).sum(dim=(1,2,3)) / mask[None,...,None].repeat(I,1,1,loss.shape[-1]).sum()

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()
    
    # calculate pae loss
    if logit_pae is not None:
        nbin = logit_pae.shape[1]
        bin_step = 0.5
        pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin-1, dtype=logit_pae.dtype, device=logit_pae.device)
        true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
        pae_loss = torch.nn.CrossEntropyLoss(reduction='none')(
            logit_pae, true_pae_label)

        pae_loss = (pae_loss * mask).sum() / (mask.sum() + eps)
    else:
        pae_loss = None
        
    return tot_loss, loss.detach(), pae_loss