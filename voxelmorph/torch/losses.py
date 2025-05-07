import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)



class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1-dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):
        
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])   # y
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])   # z
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])   # z

        #[b,c,x,y,z]
        #print ('y_pred  size ',y_pred.size())
        #print ('dy size ',dy.size())
        #print ('dx size ',dx.size())
        #print ('dz size ',dz.size())
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def JacboianDet2d(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, :, 1:, :-1] - J[:, :, :-1, :-1]
    dx = J[:, :, :-1, 1:] - J[:, :, :-1, :-1]
    Jdet2=dy[:,:,0,:] * dx[:,:,1,:] - dx[:,:,:,1] * dy[:,:,:,0]

    Jdet = Jdet2#Jdet0 - Jdet1 + Jdet2

    return Jdet

def JacboianDet(y_pred, sample_grid):
    #print ('y_pred  size ',y_pred.size())

    #print ('sample_grid  size ',sample_grid.size())
    J = y_pred + sample_grid
    #print ('J  size ',J.size())
    # the size need to be [b,c,x,y,z]
    assert (y_pred.size()[4]==3)
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    #grad = torch.autograd.grad(y_pred, sample_grid, retain_graph=True, create_graph=True)

    #print (grad)
    #print ('grad size ',grad.size())
    #print ('dy in Jacoian size ',dy.size())
    #print ('dx in Jacoian size ',dx.size())
    #print ('dz in Jacoian size ',dz.size())

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet

def JacboianLocal(y_pred, sample_grid):
    #print ('y_pred  size ',y_pred.size())

    #print ('sample_grid  size ',sample_grid.size())
    J = y_pred + sample_grid
    #print ('J  size ',J.size())
    # the size need to be [b,c,x,y,z]
    assert (y_pred.size()[4]==3)
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    #grad = torch.autograd.grad(y_pred, sample_grid, retain_graph=True, create_graph=True)

    #print (grad)
    #print ('grad size ',grad.size())
    #print ('dy in Jacoian size ',dy.size())
    #print ('dx in Jacoian size ',dx.size())
    #print ('dz in Jacoian size ',dz.size())

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet



    
def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)

def cal_cor_matrix(x,mask):
    #print (x.size())
    x=x*mask
    x=x.view(3,-1)
    cormatrix=cor_matrix(x)
    lmbda,egin=torch.eig(cormatrix)#.clip(0)
    #print(lmbda[0][0])
    #print(lmbda.item()[1][0])
    #print(lmbda.item()[2][0])

    #print (cormatrix.size())
    return cormatrix

def neg_Jdet_loss_mask(y_pred, sample_grid,mask):
    #print ('sample_grid size ',sample_grid.size())
    #cormatrix=cor_matrix(y_pred)
    #print (y_pred.size())
    #print (cormatrix.size())
    #egin=torch.eig(y_pred)#.clip(0)
    mask=mask.repeat(1,1,1,1,3)
    #print ('mask size ',mask.size())
    neg_Jdet = JacboianDet(y_pred*mask, sample_grid)
    selected_neg_Jdet = neg_Jdet##neg_JdetF.relu(neg_Jdet)

    return torch.mean((selected_neg_Jdet-1)*(selected_neg_Jdet-1))
    #return torch.mean(selected_neg_Jdet)


def compute_neg_Jdet_mask(y_pred, sample_grid,mask):
    #print ('sample_grid size ',sample_grid.size())
    #cormatrix=cor_matrix(y_pred)
    #print (y_pred.size())
    #print (cormatrix.size())
    #egin=torch.eig(y_pred)#.clip(0)
    mask=mask.repeat(1,1,1,1,3)
    #print ('mask size ',mask.size())
    neg_Jdet = JacboianDet(y_pred*mask, sample_grid)
    selected_neg_Jdet = neg_Jdet##neg_JdetF.relu(neg_Jdet)

    return torch.mean((selected_neg_Jdet)*(selected_neg_Jdet))

def second_Jdet_loss_mask(y_pred, sample_grid,mask):
    #print ('sample_grid size ',sample_grid.size())
    #cormatrix=cor_matrix(y_pred)
    #print (y_pred.size())
    #print (cormatrix.size())
    #egin=torch.eig(y_pred)#.clip(0)
    #mask=mask.repeat(1,3,1,1,1)
    #print ('mask size ',mask.size())
    neg_Jdet = JacboianDet(y_pred*mask, sample_grid)
    neg_Jdet = JacboianDet(neg_Jdet, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)

def neg_Jdet_loss2d(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet2d(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)    
    

# this allows us to pass `fweights` and `aweights`
# https://github.com/numpy/numpy/blob/623bc1fae1d47df24e7f1e29321d0c0ba2771ce0/numpy/lib/function_base.py#L2286-L2504
def cor_matrix(tensor, rowvar=True, bias=False, fweights=None, aweights=None):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)

    weights = fweights
    if aweights is not None:
        if weights is None:
            weights = aweights
        else:
            weights = weights * aweights

    if weights is None:
        mean = tensor.mean(dim=-1, keepdim=True)
    else:
        w_sum = weights.sum(dim=-1, keepdim=True)
        mean = (weights * tensor).sum(dim=-1, keepdim=True) / w_sum

    ddof = int(not bool(bias))
    if weights is None:
        fact = 1 / (tensor.shape[-1] - ddof)
    else:
        if ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            w_sum2 = (weights * aweights).sum(dim=-1, keepdim=True)
            fact = w_sum - w_sum2 / w_sum  # ddof == 1
        # warn if fact <= 0
        fact = weights / fact.relu_()

    tensor = tensor - mean
    return fact * tensor @ tensor.transpose(-1, -2).conj()


'Loss for segmentation'
import torch.nn as nn
from torch.autograd import Variable
class CrossEntropy3d_Ohem(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.9, min_kept=10000, use_weight=False):
        super(CrossEntropy3d_Ohem, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("INFO : use pre-defined weights")
            ## use the weight 
            # background   Parotid_L	Parotid_R	Submand_L	Submand_R	Mandible	Cord	BrainStem	Oral_Cav	Larynx	Chiasm	OptNrv_L	OptNrv_R	Eye_L	Eye_R

            weight = torch.FloatTensor([1, 1.5, 1.5, 2, 2, 0.9, 0.9, 0.9, 0.9, 50, 50, 30, 30, 1, 1])
            weight = torch.FloatTensor([0.5,0.57,0.57,2.30,2.25,0.24,1.05,0.55,0.14,2.86,16.17,17.34,17.60,1.86,1.83])

            # weight focus on parotid
            weight = torch.FloatTensor([0.5,2,2,2.30,2.25,0.24,1.05,0.55,0.14,2.86,16.17,17.34,17.60,1.86,1.83])
            weight = torch.FloatTensor([1,1,1,1,1,1,1,1,1,1,1,1])#,16.17,17.34,17.60,1.86,1.83])
            weight=weight.cuda()
            print (weight)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w) # good 
                target:(n, h, w) # good
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        #print (predict.size())
        #print (target.size())
        #target=target.view(target.size()[0],target.size()[2],target.size()[3])
        assert predict.dim() == 5 # good
        assert target.dim() == 4 # good
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, depth,h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:,valid_flag]
            #print('prob shape ',prob.shape)
            #print('label shape ',label.shape)
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)
        