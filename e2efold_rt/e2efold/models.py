#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.nn.modules.utils import _pair
import math
import numpy as np
from scipy.sparse import diags

from e2efold.common.utils import *
from e2efold.common.config import process_config

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride=1, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.pad = int((kernel_size-1)/2)
        self.stride = _pair(stride) 
        
    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

class ResNetblock(nn.Module):

    def __init__(self, conv, in_planes, planes, kernel_size=9, padding=8, dilation=2):
        super(ResNetblock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.bn1_2 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, planes, 
            kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(planes)
        self.bn2_2 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, 
            kernel_size=kernel_size, padding=padding, dilation=dilation)


    def forward(self, x):
        residual = x

        if len(x.shape) == 3:
            out = self.bn1(x)
        else:
            out = self.bn1_2(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        if len(out.shape) ==3:
            out = self.bn2(out)
        else:
            out = self.bn2_2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out

class ContactAttention(nn.Module):
    """
    The definition of contact network
    Instantiation: 
        d: the dimension of the hidden dimension of each base
        L: sequence length
    Input: The sequence encoding, the prior knowledge
    Output: The contact map of the input RNA sequence
    """
    def __init__(self, d, L):
        super(ContactAttention, self).__init__()
        self.d = d
        self.L = L
        # 1d convolution, L*3 to L*d
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv1d2= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn2 = nn.BatchNorm1d(d)

        self.position_embedding_1d = nn.Parameter(
            torch.randn(1, d, L)
        )

        # transformer encoder for the input sequences
        self.encoder_layer = nn.TransformerEncoderLayer(2*d, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)
        
        self.lc = LocallyConnected2d(4*d, 1, L, 1)
    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        position_embeds = self.position_embedding_1d.repeat(seq.shape[0],1,1)

        seq = torch.cat([seq, position_embeds], 1)
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # 4d*L*L

        infor = seq_mat

        contact = self.lc(infor)
        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)
    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag
        return mat

class ContactAttention_simple(nn.Module):
    """docstring for ContactAttention_simple"""
    def __init__(self, d,L):
        super(ContactAttention_simple, self).__init__()
        self.d = d
        self.L = L
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)

        self.conv_test_1 = nn.Conv2d(in_channels=6*d, out_channels=d, kernel_size=1)
        self.bn_conv_1 = nn.BatchNorm2d(d)
        self.conv_test_2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1)
        self.bn_conv_2 = nn.BatchNorm2d(d)
        self.conv_test_3 = nn.Conv2d(in_channels=d, out_channels=1, kernel_size=1)

        self.position_embedding_1d = nn.Parameter(
            torch.randn(1, d, 600)
        )

        # transformer encoder for the input sequences
        self.encoder_layer = nn.TransformerEncoderLayer(2*d, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)

    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """

        position_embeds = self.position_embedding_1d.repeat(seq.shape[0],1,1)
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq))) #d*L just for increase the capacity

        seq = torch.cat([seq, position_embeds], 1) # 2d*L
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # 4d*L*L
        
        p_mat = self.matrix_rep(position_embeds) # 2d*L*L

        infor = torch.cat([seq_mat, p_mat], 1) # 6d*L*L

        contact = F.relu(self.bn_conv_1(self.conv_test_1(infor)))
        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))
        contact = self.conv_test_3(contact)

        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)

    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag
        return mat


class ContactAttention_simple_fix_PE(ContactAttention_simple):
    """docstring for ContactAttention_simple_fix_PE"""
    def __init__(self, d, L):
        super(ContactAttention_simple_fix_PE, self).__init__(d, L)
        self.PE_net = nn.Sequential(
            nn.Linear(111,5*d),
            nn.ReLU(),
            nn.Linear(5*d,5*d),
            nn.ReLU(),
            nn.Linear(5*d,d))

        
    def forward(self, pe, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """        
        #for tup in self.PE_net.state_dict():
        #    print(tup[0], self.PE_net.state_dict()[tup].size())
        position_embeds = self.PE_net(pe.view(-1, 111)).view(-1, self.L, self.d) # N*L*111 -> N*L*d
        position_embeds = position_embeds.permute(0, 2, 1) # N*d*L
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq))) #d*L just for increase the capacity

        seq = torch.cat([seq, position_embeds], 1) # 2d*L
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # 4d*L*L
        
        p_mat = self.matrix_rep(position_embeds) # 2d*L*L

        infor = torch.cat([seq_mat, p_mat], 1) # 6d*L*L

        contact = F.relu(self.bn_conv_1(self.conv_test_1(infor)))
        contact = F.relu(self.bn_conv_2(self.conv_test_2(contact)))
        contact = self.conv_test_3(contact)

        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)
        


class ContactAttention_fix_em(nn.Module):
    """
    The definition of contact network
    Instantiation: 
        d: the dimension of the hidden dimension of each base
        L: sequence length
    Input: The sequence encoding, the prior knowledge
    Output: The contact map of the input RNA sequence
    """
    def __init__(self, d, L):
        super(ContactAttention_fix_em, self).__init__()
        # self.device = torch.device("cuda:"+config.gpu if torch.cuda.is_available() else "cpu")
        self.d = d
        self.L = L
        # 1d convolution, L*3 to L*d
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv1d2= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn2 = nn.BatchNorm1d(d)

        self.fix_pos_em_1d = torch.Tensor(np.arange(1,L+1)/np.float(L)).view(1,1,L).cuda()

        pos_j, pos_i = np.meshgrid(np.arange(1,L+1)/np.float(L), 
            np.arange(1,L+1)/np.float(L))
        self.fix_pos_em_2d = torch.cat([torch.Tensor(pos_i).unsqueeze(0), 
            torch.Tensor(pos_j).unsqueeze(0)], 0).unsqueeze(0).cuda()

        # transformer encoder for the input sequences
        self.encoder_layer = nn.TransformerEncoderLayer(d+1, 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 3)

        self.lc = LocallyConnected2d(2*d+2+2, 1, L, 1)
        self.conv_test = nn.Conv2d(in_channels=2*d+2+2, out_channels=1, 
            kernel_size=1)

    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        position_embeds = self.fix_pos_em_1d.repeat(seq.shape[0],1,1)

        seq = torch.cat([seq, position_embeds], 1)
        seq = self.transformer_encoder(seq.permute(-1, 0, 1))
        seq = seq.permute(1, 2, 0)

        # what about apply attention on the the 2d map?
        seq_mat = self.matrix_rep(seq) # (2d+2)*L*L

        position_embeds_2d = self.fix_pos_em_2d.repeat(seq.shape[0],1,1,1)
        infor = torch.cat([seq_mat, position_embeds_2d], 1) #(2d+2+2)*L*L

        contact = self.lc(infor)
        # contact = self.conv_test(infor)
        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2

        return contact.view(-1, self.L, self.L)



    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat


class Lag_PP_NN(nn.Module):
    """
    The definition of Lagrangian post-procssing with neural network parameterization
    Instantiation: 
    :steps: the number of unroll steps
    Input: 
    :u: the utility matrix, batch*L*L
    :s: the sequence encoding, batch*L*4

    Output: a list of contact map of each step, batch*L*L
    """
    def __init__(self, steps, k):
        super(Lag_PP_NN, self).__init__()
        self.steps = steps
        # the parameter for the soft sign
        # the k value need to be tuned
        self.k = k
        self.s = math.log(9.0)
        self.w = 1
        self.rho = 1
        # self.s = nn.Parameter(torch.randn(1))
        # self.w = nn.Parameter(torch.randn(1))
        # self.a_hat_conv_list = nn.ModuleList()
        # self.rho_conv_list = nn.ModuleList()
        # self.lmbd_conv_list = nn.ModuleList()
        # self.make_update_cnns(steps)

        self.a_hat_fc_list = nn.ModuleList()
        self.rho_fc_list = nn.ModuleList()
        self.lmbd_fc_list = nn.ModuleList()        
        self.make_update_fcs(steps)

    def make_update_fcs(self, steps):
        for i in range(steps):
            a_hat_fc_tmp = nn.Sequential(
                nn.Linear(3,3),
                nn.ReLU(),
                nn.Linear(3,1),
                nn.ReLU())
            rho_fc_tmp = nn.Sequential(
                nn.Linear(3,3),
                nn.ReLU(),
                nn.Linear(3,1),
                nn.ReLU())
            lmbd_fc_tmp = nn.Sequential(
                nn.Linear(2,3),
                nn.ReLU(),
                nn.Linear(3,1),
                nn.ReLU())
            self.a_hat_fc_list.append(a_hat_fc_tmp)
            self.rho_fc_list.append(rho_fc_tmp)
            self.lmbd_fc_list.append(lmbd_fc_tmp)

    def make_update_cnns(self, steps):
        for i in range(steps):
            a_hat_conv_tmp = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
                nn.ReLU())
            rho_conv_tmp = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
                nn.ReLU())
            lmbd_conv_tmp = nn.Sequential(
                nn.Conv1d(in_channels=2, out_channels=3, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
                nn.ReLU())
            self.a_hat_conv_list.append(a_hat_conv_tmp)
            self.rho_conv_list.append(rho_conv_tmp)
            self.lmbd_conv_list.append(lmbd_conv_tmp)

    def forward(self, u, x, timesteps):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k)
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = self.w*F.relu(torch.sum(a_tmp, dim=-1) - 1)

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(timesteps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule_fc(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule_fc(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        # grad: n*L*L

        # reshape them first: N*L*L*3 => NLL*3
        a_hat_fc = self.a_hat_fc_list[t]
        rho_fc = self.rho_fc_list[t]
        input_features = torch.cat([torch.unsqueeze(a_hat,-1),
            torch.unsqueeze(grad,-1), torch.unsqueeze(u,-1)], -1).view(-1, 3)
        a_hat_updated = a_hat_fc(input_features).view(a_hat.shape)  

        rho = rho_fc(input_features).view(a_hat.shape) # rho直接从input_feature里学，用RT之后如何反向传播？
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        # a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho)
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        # lmbd: n*L, so we should use 1d conv
        lmbd_fc = self.lmbd_fc_list[t]
        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_input_features = torch.cat([torch.unsqueeze(lmbd, -1),
            torch.unsqueeze(lmbd_grad, -1)], -1).view(-1, 2)
        lmbd_updated = lmbd_fc(lmbd_input_features).view(lmbd.shape)

        return lmbd_updated, a_updated, a_hat_updated




    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        # grad: n*L*L

        # we update the a_hat with 2 conv layers whose filters are 1 by 1
        # so that different positions can share parameters
        # we put a_hat, g and u as three channels and the output a_hat as one channel
        # the inputs are N*3*L*L
        a_hat_conv = self.a_hat_conv_list[t]
        rho_conv = self.rho_conv_list[t]
        input_features = torch.cat([torch.unsqueeze(a_hat,1),
            torch.unsqueeze(grad,1), torch.unsqueeze(u,1)], 1)
        a_hat_updated = torch.squeeze(a_hat_conv(input_features), 1)
        # rho = torch.squeeze(rho_conv(input_features),1)
        # a_hat_updated = F.relu(torch.abs(a_hat) - rho)
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho)
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        # lmbd: n*L, so we should use 1d conv
        lmbd_conv = self.lmbd_conv_list[t]
        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_input_features = torch.cat([torch.unsqueeze(lmbd,1),
            torch.unsqueeze(lmbd_grad,1)], 1)
        lmbd_updated = torch.squeeze(lmbd_conv(lmbd_input_features), 1)

        return lmbd_updated, a_updated, a_hat_updated

    def constraint_matrix_batch(self, x):
        base_a = x[:, :, 0]
        base_u = x[:, :, 1]
        base_c = x[:, :, 2]
        base_g = x[:, :, 3]
        batch = base_a.shape[0]
        length = base_a.shape[1]
        au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
        au_ua = au + torch.transpose(au, -1, -2)
        cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
        cg_gc = cg + torch.transpose(cg, -1, -2)
        ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
        ug_gu = ug + torch.transpose(ug, -1, -2)
        return au_ua + cg_gc + ug_gu
    
    def contact_a(self, a_hat, m):
        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m
        return a


class Lag_PP_zero(nn.Module):
    """
    The definition of Lagrangian post-procssing with no parameters
    Instantiation: 
    :steps: the number of unroll steps
    Input: 
    :u: the utility matrix, batch*L*L
    :s: the sequence encoding, batch*L*4

    Output: a list of contact map of each step, batch*L*L
    """
    def __init__(self, steps, k):
        super(Lag_PP_zero, self).__init__()
        #self.device = torch.device("cuda:"+config.gpu if torch.cuda.is_available() else "cpu")
        self.steps = steps
        # the parameter for the soft sign
        self.k = k
        self.s = math.log(9.0)
        self.rho = 1.0
        self.alpha = 0.01
        self.beta = 0.1
        self.lr_decay = 0.99

    def forward(self, u, x, timesteps):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(timesteps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * grad
        self.alpha *= self.lr_decay
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho*self.alpha)
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * lmbd_grad
        self.beta *= self.lr_decay
        
        return lmbd_updated, a_updated, a_hat_updated

    def constraint_matrix_batch(self, x):
        base_a = x[:, :, 0]
        base_u = x[:, :, 1]
        base_c = x[:, :, 2]
        base_g = x[:, :, 3]
        batch = base_a.shape[0]
        length = base_a.shape[1]
        au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
        au_ua = au + torch.transpose(au, -1, -2)
        cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
        cg_gc = cg + torch.transpose(cg, -1, -2)
        ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
        ug_gu = ug + torch.transpose(ug, -1, -2)
        m = au_ua + cg_gc + ug_gu

        mask = diags([1]*7, [-3, -2, -1, 0, 1, 2, 3], 
            shape=(m.shape[-2], m.shape[-1])).toarray()
        m = m.masked_fill(torch.Tensor(mask).bool().cuda(), 0)#.to(self.device), 0)
        return m
    
    def contact_a(self, a_hat, m):
        a = a_hat * a_hat
        a = (a + torch.transpose(a, -1, -2)) / 2
        a = a * m
        return a


class Lag_PP_perturb(Lag_PP_zero):
    def __init__(self, steps, k):
        super(Lag_PP_perturb, self).__init__(steps, k)
        self.steps = steps
        self.k = k
        self.lr_decay = nn.Parameter(torch.Tensor([0.99]))
        # self.s = nn.Parameter(torch.Tensor([math.log(9.0)]))
        self.s = math.log(9.0)
        self.rho = nn.ParameterList([nn.Parameter(torch.Tensor([1.0])) for i in range(steps)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.Tensor([0.01*math.pow(self.lr_decay, 
            i)])) for i in range(steps)])
        self.beta = nn.ParameterList([nn.Parameter(torch.Tensor([0.1*math.pow(self.lr_decay, 
            i)])) for i in range(steps)])
        

    def forward(self, u, x, timesteps):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(timesteps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha[t] * grad
        a_hat_updated = F.relu(torch.abs(a_hat_updated) - self.rho[t]*self.alpha[t])
        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta[t] * lmbd_grad
        
        return lmbd_updated, a_updated, a_hat_updated


class Lag_PP_mixed(Lag_PP_zero):
    """
    For the update of a and lambda, we use gradient descent with 
    learnable parameters. For the rho, we use neural network to learn
    a position related threshold
    """
    def __init__(self, steps, k, rho_mode='fix'):
        super(Lag_PP_mixed, self).__init__(steps, k)
        #self.device = torch.device("cuda:"+config.gpu if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.k = k
        # self.s = nn.Parameter(torch.ones(600, 600)*math.log(9.0))
        self.s = nn.Parameter(torch.Tensor([math.log(9.0)]))
        # self.s = math.log(9.0)
        self.w = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.Tensor([1.0]))
        # self.rho = 1.0
        self.rho_m = nn.Parameter(torch.randn(600, 600))
        self.rho_net = nn.Sequential(
                nn.Linear(3,5),
                nn.ReLU(),
                nn.Linear(5,1),
                nn.ReLU())
        # build the rho network
        # reuse it under every time step
        self.alpha = nn.Parameter(torch.Tensor([0.01]))
        self.beta = nn.Parameter(torch.Tensor([0.1]))
        self.lr_decay_alpha = nn.Parameter(torch.Tensor([0.99]))
        self.lr_decay_beta = nn.Parameter(torch.Tensor([0.99]))
        # self.alpha = torch.Tensor([0.01]).cuda()
        # self.beta = torch.Tensor([0.1]).cuda()
        # self.lr_decay_alpha = torch.Tensor([0.99]).cuda()
        # self.lr_decay_beta = torch.Tensor([0.99]).cuda()
        self.rho_mode = rho_mode

        pos_j, pos_i = np.meshgrid(np.arange(1,600+1)/600.0, 
            np.arange(1,600+1)/600.0)
        self.rho_pos_fea = torch.cat([torch.Tensor(pos_i).unsqueeze(-1), 
            torch.Tensor(pos_j).unsqueeze(-1)], -1).view(-1, 2).cuda()#.to(self.device)

        self.rho_pos_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU()
            )

    def forward(self, u, x, timesteps):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = self.w * F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(timesteps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]

    def update_rule(self, u, m, lmbd, a, a_hat, t):

        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * torch.pow(self.lr_decay_alpha,
            t) * grad
        # the rho needs to be further dealt

        if self.rho_mode=='nn':
            input_features = torch.cat([torch.unsqueeze(a_hat,-1),
                torch.unsqueeze(grad,-1), torch.unsqueeze(u,-1)], -1).view(-1, 3)
            rho = self.rho_net(input_features).view(a_hat.shape)
            a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        elif self.rho_mode=='matrix':
            a_hat_updated = F.relu(
                torch.abs(a_hat_updated) - self.rho_m*self.alpha * torch.pow(self.lr_decay_alpha,t))
        elif self.rho_mode=='nn_pos':
            rho = self.rho_pos_net(self.rho_pos_fea).view(
                a_hat_updated.shape[-2], a_hat_updated.shape[-1])
            a_hat_updated = F.relu(torch.abs(a_hat_updated) - rho)
        else:
            a_hat_updated = F.relu(
                torch.abs(a_hat_updated) - self.rho*self.alpha * torch.pow(self.lr_decay_alpha,t))           

        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * torch.pow(self.lr_decay_beta, 
            t) * lmbd_grad
        
        return lmbd_updated, a_updated, a_hat_updated
        
class Lag_PP_final(Lag_PP_zero):
    """
    For the update of a and lambda, we use gradient descent with 
    learnable parameters. For the rho, we use neural network to learn
    a position related threshold
    """
    def __init__(self, steps, k, rho_mode='fix'):
        super(Lag_PP_final, self).__init__(steps, k)
        #self.device = torch.device("cuda:"+config.gpu if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.k = k
        self.s = nn.Parameter(torch.Tensor([math.log(9.0)]))
        self.w = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.Tensor([1.0]))
        # build the rho network
        # reuse it under every time step
        self.alpha = nn.Parameter(torch.Tensor([0.01]))
        self.beta = nn.Parameter(torch.Tensor([0.1]))
        self.lr_decay_alpha = nn.Parameter(torch.Tensor([0.99]))
        self.lr_decay_beta = nn.Parameter(torch.Tensor([0.99]))
        self.rho_mode = rho_mode

    
    def forward(self, u, x, timesteps):
        a_t_list = list()
        a_hat_t_list = list()
        lmbd_t_list = list()

        m = self.constraint_matrix_batch(x) # N*L*L

        u = soft_sign(u - self.s, self.k) * u

        # initialization
        a_hat_tmp = (torch.sigmoid(u)) * soft_sign(u - self.s, self.k).detach()
        a_tmp = self.contact_a(a_hat_tmp, m)
        lmbd_tmp = self.w * F.relu(torch.sum(a_tmp, dim=-1) - 1).detach()

        lmbd_t_list.append(lmbd_tmp)
        a_t_list.append(a_tmp)
        a_hat_t_list.append(a_hat_tmp)
        # gradient descent
        for t in range(timesteps):
            lmbd_updated, a_updated, a_hat_updated = self.update_rule(
                u, m, lmbd_tmp, a_tmp, a_hat_tmp, t)

            a_hat_tmp = a_hat_updated
            a_tmp = a_updated
            lmbd_tmp = lmbd_updated

            lmbd_t_list.append(lmbd_tmp)
            a_t_list.append(a_tmp)
            a_hat_t_list.append(a_hat_tmp)

        # return a_updated
        return a_t_list[1:]
    

    def update_rule(self, u, m, lmbd, a, a_hat, t):
        grad_a = - u / 2 + (lmbd * soft_sign(torch.sum(a,
            dim=-1) - 1, self.k)).unsqueeze_(-1).expand(u.shape)
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))

        a_hat_updated = a_hat - self.alpha * torch.pow(self.lr_decay_alpha,
            t) * grad
        # the rho needs to be further dealt
        a_hat_updated = F.relu(
            torch.abs(a_hat_updated) - self.rho*self.alpha * torch.pow(self.lr_decay_alpha,t))           

        a_hat_updated = torch.clamp(a_hat_updated, -1, 1)
        a_updated = self.contact_a(a_hat_updated, m)

        lmbd_grad = F.relu(torch.sum(a_updated, dim=-1) - 1)
        lmbd_updated = lmbd + self.beta * torch.pow(self.lr_decay_beta, 
            t) * lmbd_grad
        
        return lmbd_updated, a_updated, a_hat_updated

class RNA_SS_e2e(nn.Module):
    def __init__(self, model_att, model_pp):
        super(RNA_SS_e2e, self).__init__()
        self.model_att = model_att
        self.model_pp = model_pp
        
    def forward(self, prior, seq, state, timesteps):
        u = self.model_att(prior, seq, state)
        map_list = self.model_pp(u, seq, timesteps)
        return u, map_list

# get args 
args = get_args()

config_file = args.config

config = process_config(config_file)
print("#####Stage 3#####")
print('Here is the configuration of this run: ')
print(config)

os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
#device = torch.device("cuda:"+config.gpu if torch.cuda.is_available() else "cpu")
# print("gpu config "+config.gpu)

d = config.u_net_d
BATCH_SIZE = config.BATCH_SIZE
OUT_STEP = config.OUT_STEP
LOAD_MODEL = config.LOAD_MODEL
pp_steps = config.pp_steps
pp_loss = config.pp_loss
data_type = config.data_type
model_type = config.model_type
pp_type = '{}_s{}'.format(config.pp_model, pp_steps)
rho_per_position = config.rho_per_position
model_path = '../models_ckpt/supervised_{}_{}_d{}_l3_upsampling.pt'.format(model_type, data_type,d)
pp_model_path = '../models_ckpt/lag_pp_{}_{}_{}_position_{}.pt'.format(
    pp_type, data_type, pp_loss,rho_per_position)
# The unrolled steps for the upsampling model is 10
# e2e_model_path = '../models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}_upsampling.pt'.format(model_type,
#     pp_type,d, data_type, pp_loss,rho_per_position)
e2e_model_path = '../models_ckpt/e2e_{}_{}_d{}_{}_{}_position_{}.pt'.format(model_type,
    pp_type,d, data_type, pp_loss,rho_per_position)
epoches_third = config.epoches_third
evaluate_epi = config.evaluate_epi
step_gamma = config.step_gamma
k = config.k

# for loading data
# loading the rna ss data, the data has been preprocessed
# 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
from e2efold.data_generator import RNASSDataGenerator, Dataset
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 
    'seq ss_label length name pairs')

#train_data = RNASSDataGenerator('/data/chenzhijie/rna/data/{}/'.format(data_type), 'train', True)
#val_data = RNASSDataGenerator('/data/chenzhijie/rna/data/{}/'.format(data_type), 'val')
# test_data = RNASSDataGenerator('../data/{}/'.format(data_type), 'test_no_redundant')
test_data = RNASSDataGenerator('/mnt/Datawarehouse/chenzhijie2/data/rnastralign_all/', 'test_no_redundant_600')
train_data = test_data
val_data = test_data

seq_len = train_data.data_y.shape[-2]
print('Max seq length ', seq_len)

# using the pytorch interface to parallel the data generation and model training
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 1,
          'drop_last': True}
train_set = Dataset(train_data)
train_generator = data.DataLoader(train_set, **params)

val_set = Dataset(val_data)
val_generator = data.DataLoader(val_set, **params)

#test_set = Dataset(test_data)
#test_generator = data.DataLoader(test_set, **params)

# only for save the final results

params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 1,
          'drop_last': False}
test_set = Dataset(test_data)
test_generator = data.DataLoader(test_set, **params)

# define the model
print("—---------------------")
print(model_type)
print("----------------------")
if model_type =='test_lc':
    contact_net = ContactNetwork_test(d=d, L=seq_len).cuda()#.to(device)
if model_type == 'att6':
    contact_net = ContactAttention(d=d, L=seq_len).cuda()#.to(device)
if model_type == 'att_simple':
    contact_net = ContactAttention_simple(d=d, L=seq_len).cuda()#.to(device)    
if model_type == 'att_simple_fix':
    contact_net = ContactAttention_simple_fix_PE(d=d, L=seq_len).cuda()#.to(device)
if model_type == 'fc':
    contact_net = ContactNetwork_fc(d=d, L=seq_len).cuda()#.to(device)
if model_type == 'conv2d_fc':
    contact_net = ContactNetwork(d=d, L=seq_len).cuda()#.to(device)

# need to write the class for the computational graph of lang pp
if pp_type=='nn':
    lag_pp_net = Lag_PP_NN(pp_steps, k).cuda()#.to(device)
if 'zero' in pp_type:
    lag_pp_net = Lag_PP_zero(pp_steps, k).cuda()#.to(device)
if 'perturb' in pp_type:
    lag_pp_net = Lag_PP_perturb(pp_steps, k).cuda()#.to(device)
if 'mixed'in pp_type:
    lag_pp_net = Lag_PP_mixed(pp_steps, k, rho_per_position).cuda()#.to(device)

if LOAD_MODEL and os.path.isfile(model_path):
    print('Loading u net model...')
    contact_net.load_state_dict(torch.load(model_path))
if LOAD_MODEL and os.path.isfile(pp_model_path):
    print('Loading pp model...')
    lag_pp_net.load_state_dict(torch.load(pp_model_path))

rna_ss_e2e = RNA_SS_e2e(contact_net, lag_pp_net)

if LOAD_MODEL and os.path.isfile(e2e_model_path):
    print('Loading e2e model...')
    rna_ss_e2e.load_state_dict(torch.load(e2e_model_path))

def assign_params(model, params):
    static_named_parameters = []
    for n_and_p in model.named_parameters():
        static_named_parameters.append(n_and_p)
    for name_and_param, new_param in zip(
            static_named_parameters, params):
        name, old_param = name_and_param
        #if 'model_att' in name:
        #    setattr(model.model_att, 'weight', new_param)
        #if name == 'encoder.weight' and FLAGS.tied:
        #    setattr(model.decoder, 'weight', new_param)
        #if name == 'decoder.weight' and FLAGS.tied:
        #    pdb.set_trace()
        module = model
        while len(name.split('.')) > 1:
            component_name = name.split('.')[0]
            module = getattr(module, component_name)
            name = '.'.join(name.split('.')[1:])
        #print("--------------")
        assert(old_param.size() == new_param.size())
        #if (old_param.size() != new_param.size()):
        #with open("output.txt", "a") as txt_file:
        #    txt_file.write(str(module) + " " + name + " " + \
        #         str(old_param.size()) + " " + str(new_param.size()) + "\n")
        #print(module, name, old_param.size(), new_param.size())
        setattr(module, name, new_param)
    with open("output_model_name.txt", "a") as txt_file:
        for name_and_param in static_named_parameters:
            name, old_param = name_and_param
            txt_file.write(name + "\n" + str(old_param.data.size()) + "\n")


def eval_fn(params, test_generator, timesteps):
    #for param in params:
    #    print(param.shape)
    print("*******************")
    print("Evaluating...")
    print("*******************")
    assign_params(rna_ss_e2e, params)
    rna_ss_e2e.eval()
    total_loss = 0

    device = torch.device("cuda:"+config.gpu if torch.cuda.is_available() else "cpu")
    #contact_net.eval()
    #lag_pp_net.eval()
    result_no_train = list()
    result_no_train_shift = list()
    result_pp = list()
    result_pp_shift = list()

    f1_no_train = list()
    f1_pp = list()
    seq_lens_list = list()

    batch_n = 0
    for contacts, seq_embeddings, matrix_reps, seq_lens in test_generator:
        if batch_n == 20:
            break
        if batch_n %10==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).cuda()#.to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).cuda()#.to(device)
        matrix_reps_batch = torch.unsqueeze(
            torch.Tensor(matrix_reps.float()).cuda(), -1)

        state_pad = torch.zeros(contacts.shape).cuda()#.to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().cuda()#.to(device)
        with torch.no_grad():
            pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, # prior
                            seq_embedding_batch, state_pad, timesteps) # seq, state

        # the learning pp result
        final_pred = (a_pred_list[-1].cpu()>0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp += result_tmp

        result_tmp_shift = list(map(lambda i: evaluate_shifted(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_pp_shift += result_tmp_shift

        f1_tmp = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        f1_pp += f1_tmp
        seq_lens_list += list(seq_lens)

    pp_exact_p,pp_exact_r,pp_exact_f1 = zip(*result_pp)
    pp_shift_p,pp_shift_r,pp_shift_f1 = zip(*result_pp_shift)  
    print('Average testing F1 score with learning post-processing: ', np.average(pp_exact_f1))
    print('Average testing F1 score with learning post-processing allow shift: ', np.average(pp_shift_f1))

    print('Average testing precision with learning post-processing: ', np.average(pp_exact_p))
    print('Average testing precision with learning post-processing allow shift: ', np.average(pp_shift_p))

    print('Average testing recall with learning post-processing: ', np.average(pp_exact_r))
    print('Average testing recall with learning post-processing allow shift: ', np.average(pp_shift_r))

    e2e_result_df = pd.DataFrame()
    e2e_result_df['name'] = [a.name for a in test_data.data[:20]]
    e2e_result_df['type'] = list(map(lambda x: x.split('/')[2], [a.name for a in test_data.data[:20]]))
    #e2e_result_df['seq_lens'] = list(map(lambda x: x.numpy(), seq_lens_list))
    e2e_result_df['exact_p'] = pp_exact_p
    e2e_result_df['exact_r'] = pp_exact_r
    e2e_result_df['exact_f1'] = pp_exact_f1
    e2e_result_df['shift_p'] = pp_shift_p
    e2e_result_df['shift_r'] = pp_shift_r
    e2e_result_df['shift_f1'] = pp_shift_f1
    for rna_type in e2e_result_df['type'].unique():
        print(rna_type)
        df_temp = e2e_result_df[e2e_result_df.type==rna_type]
        to_output = list(map(str, 
            list(df_temp[['exact_p', 'exact_r', 'exact_f1', 'shift_p','shift_r', 'shift_f1']].mean().values.round(3))))
        print(to_output)
    
    return {
        'test f1': np.average(pp_exact_f1),
        'test precision': np.average(pp_exact_p),
        'test recall': np.average(pp_exact_r)
    }


from torch.autograd import Variable
def train_fn(state, params, inputs, config, timesteps, steps_done):
    print("steps:{}".format(timesteps))
    PE_batch, seq_embedding_batch, state_pad, contact_masks, contacts_batch = inputs  # retrieve inputs
    #for param in params:
    #    print(param.shape)
    assign_params(rna_ss_e2e, params) # restore the model

    rna_ss_e2e.train()

    pred_contacts, a_pred_list = rna_ss_e2e(PE_batch, # prior
            seq_embedding_batch, state_pad, timesteps) # seq, state

    avg_loss = test_net(pred_contacts, a_pred_list, contact_masks, contacts_batch, config, timesteps, steps_done)

    compute = timesteps
    """
    avg_loss.backward()
    static_named_parameters = []
    for n_and_p in rna_ss_e2e.named_parameters():
        static_named_parameters.append(n_and_p)

    with open("output_model_name_loss.txt", "a") as txt_file:
        for name_and_param in static_named_parameters:
            name, old_param = name_and_param
            txt_file.write(name + " " + str(old_param.data.shape))
            if old_param.grad is not None:
                txt_file.write(str(old_param.grad.shape) + "\n")
            else:
                txt_file.write("??????\n")
    sys.exit()
    """
    return avg_loss, compute

pos_weight = torch.Tensor([300]).cuda()#.to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
    pos_weight = pos_weight)
criterion_mse = torch.nn.MSELoss(reduction='sum')

def test_net(pred_contacts, a_pred_list, contact_masks, contacts_batch, config, timesteps, steps_done): 
    #pp_steps = config.pp_steps
    pp_steps = timesteps
    pp_loss = config.pp_loss
    step_gamma = config.step_gamma

    loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
    # Compute loss, consider the intermidate output
    if pp_loss == "l2":
        loss_a = criterion_mse(
            a_pred_list[-1]*contact_masks, contacts_batch)
        for i in range(pp_steps-1):
            loss_a += np.power(step_gamma, pp_steps-1-i)*criterion_mse(
                a_pred_list[i]*contact_masks, contacts_batch)
        mse_coeff = 1.0/(seq_len*pp_steps)

    if pp_loss == 'f1':
        loss_a = f1_loss(a_pred_list[-1]*contact_masks, contacts_batch)
        for i in range(pp_steps-1):
            loss_a += np.power(step_gamma, pp_steps-1-i)*f1_loss(
                a_pred_list[i]*contact_masks, contacts_batch)            
        mse_coeff = 1.0/pp_steps

    loss_a = mse_coeff*loss_a

    loss = loss_u + loss_a

    if steps_done % OUT_STEP ==0:
        print('Stage 3, step: {}, loss_u: {}, loss_a: {}, loss: {}'.format(
            steps_done, loss_u, loss_a, loss))

        final_pred = a_pred_list[-1].cpu()>0.5
        f1 = list(map(lambda i: F1_low_tri(final_pred.cpu()[i], 
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        print('Average training F1 score: ', np.average(f1))
    return loss


# only using convolutional layers is problematic
# Indeed, if we only use CNN, the spatial information is missing
class ContactNetwork(nn.Module):
    """
    The definition of contact network
    Instantiation: 
        d: the dimension of the hidden dimension of each base
        L: sequence length
    Input: The sequence encoding, the prior knowledge
    Output: The contact map of the input RNA sequence
    """
    def __init__(self, d, L):
        super(ContactNetwork, self).__init__()
        self.d = d
        self.L = L
        # 1d convolution, L*3 to L*d
        self.conv1d1= nn.Conv1d(in_channels=4, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn1 = nn.BatchNorm1d(d)
        self.conv1d2= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn2 = nn.BatchNorm1d(d)

        # 2d convolution for the matrix representation
        # if possible, we may think of make dilation related the the sequence length
        # and we can consider short-cut link
        self.conv2d1 = nn.Conv2d(in_channels=2*d, out_channels=4*d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn3 = nn.BatchNorm2d(4*d)
        self.conv2d2 = nn.Conv2d(in_channels=4*d, out_channels=2*d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn4 = nn.BatchNorm2d(2*d)

        # 2d convolution for the state
        self.conv2d3 = nn.Conv2d(in_channels=1, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn5 = nn.BatchNorm2d(d)
        self.conv2d4 = nn.Conv2d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn6 = nn.BatchNorm2d(d)

        # final convolutional and global pooling, as well as the fc net
        # we may think about multiple paths
        self.conv1 = nn.Conv2d(in_channels=2*d+3, out_channels=3*d, 
            kernel_size=20, padding=19, dilation=2)
        self.bn7 = nn.BatchNorm2d(3*d)
        self.conv2 = nn.Conv2d(in_channels=3*d, out_channels=3*d, 
            kernel_size=20, padding=19, dilation=2)
        self.bn8 = nn.BatchNorm2d(3*d)
        self.conv3 = nn.Conv2d(in_channels=3*d, out_channels=1, 
            kernel_size=20, padding=19, dilation=2)

        self.fc1 = nn.Linear(L*L, L*L)
       

    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        seq_mat = self.matrix_rep(seq) # 2d*L*L
        seq_mat = F.relu(self.bn3(self.conv2d1(seq_mat)))
        seq_mat = F.relu(self.bn4(self.conv2d2(seq_mat))) # 2d*L*L

        state = nn.functional.one_hot(state.to(torch.int64)-state.min(), 3) # L*L*3
        state = state.permute(0, 3, 1, 2).to(torch.float32) # 3*L*L

        # prior = prior.permute(0, 3, 1, 2).to(torch.float32) # 1*L*L
        # prior = F.relu(self.bn5(self.conv2d3(prior)))
        # prior = F.relu(self.bn6(self.conv2d4(prior))) # d*L*L

        infor = torch.cat([seq_mat, state], 1) # (3d+3)*L*L
        infor = F.relu(self.bn7(self.conv1(infor)))
        # infor = F.relu(self.bn8(self.conv2(infor))) # 3d*L*L
        infor = F.relu(self.conv3(infor)) #1*L*L

        # final dense net
        contact = self.fc1(infor.view(-1, self.L*self.L))
        # contact = infor

        return contact.view(-1, self.L, self.L)
        # return torch.squeeze(infor, 1)


    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1) # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L,1,1)
        x2 = x2.repeat(1, 1, L,1)
        mat = torch.cat([x,x2],-1) # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2)) # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat

class ContactNetwork_test(ContactNetwork):
    def __init__(self, d, L):
        super(ContactNetwork_test, self).__init__(d,L)
        self.resnet1d = self._make_layer(ResNetblock, nn.Conv1d, 4, d)
        self.resnet1d_2 = self._make_layer(ResNetblock, nn.Conv1d, 4, d)
        # self.fc1 = nn.Linear(self.d*self.L, self.L*self.L)

        self.conv1d3= nn.Conv1d(in_channels=d+L, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn3 = nn.BatchNorm1d(d)
        self.conv1d4= nn.Conv1d(in_channels=d, out_channels=d, 
            kernel_size=9, padding=8, dilation=2)
        self.bn4 = nn.BatchNorm1d(d)


        self.conv_test = nn.Conv2d(in_channels=3*d, out_channels=1, 
            kernel_size=9, padding=8, dilation=2)
        self.bn_test = nn.BatchNorm2d(1)

        self.position_embedding = nn.Parameter(
            torch.randn(1, d, L, L)
        )

        self.lc = LocallyConnected2d(2*d, 1, L, 1)

    def _make_layer(self, block, conv, layers, plane):
        l = []
        for i in range(layers):
            l.append(block(conv, plane, plane))
        return nn.Sequential(*l)

    def forward(self, prior, seq, state):
        """
        state: L*L*1
        seq: L*4
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        infor = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        infor = self.resnet1d(infor) # d*L

        infor = self.matrix_rep(infor) # 2d*L*L

        # position_embeds = self.position_embedding.repeat(infor.shape[0],1,1,1)

        # infor = torch.cat([infor, position_embeds], 1)

        # prior = torch.squeeze(prior, -1)
        # infor = torch.cat([prior, infor], 1) # (d+L)*L
        # infor = F.relu(self.bn3(self.conv1d3(infor)))
        # infor = self.resnet1d_2(infor) # d*L
        # contact = self.fc1(infor.view(-1, self.d*self.L))
        # contact = self.bn_test(self.conv_test(infor))
        contact = self.lc(infor)
        contact = contact.view(-1, self.L, self.L)
        contact = (contact+torch.transpose(contact, -1, -2))/2


        return contact.view(-1, self.L, self.L)




class ContactNetwork_fc(ContactNetwork_test):
    """docstring for ContactNetwork_fc"""
    def __init__(self, d, L):
        super(ContactNetwork_fc, self).__init__(d, L)
        self.fc1 = nn.Linear(self.d*self.L, self.L*self.L)

    def forward(self, prior, seq, state):
        """
        state: L*L*1
        seq: L*4
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        infor = F.relu(self.bn2(self.conv1d2(seq))) # d*L
        infor = self.resnet1d(infor) # d*L

        # infor = self.matrix_rep(infor) # 2d*L*L

        # position_embeds = self.position_embedding.repeat(infor.shape[0],1,1,1)

        # infor = torch.cat([infor, position_embeds], 1)

        # prior = torch.squeeze(prior, -1)
        # infor = torch.cat([prior, infor], 1) # (d+L)*L
        # infor = F.relu(self.bn3(self.conv1d3(infor)))
        # infor = self.resnet1d_2(infor) # d*L
        contact = self.fc1(infor.view(-1, self.d*self.L))
        contact = contact.view(-1, self.L, self.L)
        
        # contact = (contact+torch.transpose(contact, -1, -2))/2


        return contact.view(-1, self.L, self.L)

# need to further add the prior knowledge block and the state embedding
class ContactNetwork_ResNet(ContactNetwork):
    def __init__(self, d, L):
        super(ContactNetwork_ResNet, self).__init__(d,L)
        self.resnet1d = self._make_layer(ResNetblock, nn.Conv1d, 4, d)
        self.resnet2d = self._make_layer(ResNetblock, nn.Conv2d, 4, 3*d)
        self.fc1 = nn.Linear(L*L, L*L)
        self.dropout = nn.Dropout(p=0.2)
        self.lc = LocallyConnected2d(3*d, 1, L, 5)

    def _make_layer(self, block, conv, layers, plane):
        l = []
        for i in range(layers):
            l.append(block(conv, plane, plane))
        return nn.Sequential(*l)


    def forward(self, prior, seq, state):
        """
        prior: L*L*1
        seq: L*4
        state: L*L
        """
        seq = seq.permute(0, 2, 1) # 4*L
        seq = F.relu(self.bn1(self.conv1d1(seq)))
        seq = self.resnet1d(seq) # d*L
        seq_mat = self.matrix_rep(seq) # 2d*L*L

        # deal with state, first embed state
        state = nn.functional.one_hot(state.to(torch.int64)-state.min(), 3) # L*L*3
        state = state.permute(0, 3, 1, 2).to(torch.float32) # 3*L*L

        # prior = prior.permute(0, 3, 1, 2).to(torch.float32) # 1*L*L
        # prior = F.relu(self.bn5(self.conv2d3(prior)))
        # prior = F.relu(self.bn6(self.conv2d4(prior))) # d*L*L

        infor = torch.cat([seq_mat, state], 1) # (2d+3)*L*L
        infor = F.relu(self.bn7(self.conv1(infor)))
        # infor = F.relu(self.bn8(self.conv2(infor))) # 3d*L*L
        infor = self.resnet2d(infor) # 3d*L*L

        # final dense net
        infor = F.relu(self.conv3(infor)) #1*L*L
        # contact = self.fc1(self.dropout(infor.view(-1, self.L*self.L)))
        contact = infor
        # the final locally connected net
        # contact = self.lc(infor)

        return contact.view(-1, self.L, self.L)

# for testing
def testing():
    seq = torch.rand([32, 135, 4])
    contact = torch.zeros([32, 135,135,1], dtype=torch.int32)
    contact[:, :,0]=1
    contact[:, :,1]=-1
    state = torch.zeros([32, 135, 135], dtype=torch.int32)
    m = ContactNetwork_ResNet(d=3, L=135)
    contacts = m(contact, seq, state)
    return contacts
