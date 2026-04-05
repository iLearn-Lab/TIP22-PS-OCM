from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import torchvision 
import copy,sys
import json
import math
import string 
from torch.nn.parameter import Parameter


class Image_net(nn.Module):
    def __init__(self, embedding_dim, outfit_threshold, att_num_dic={}):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Linear(512, embedding_dim)

        self.out_size = 64
        self.att_num = 12
        self.hidden_dim = embedding_dim
        self.outfit_threshold = outfit_threshold
        #===========================  disentangle attribute  =============================
        self.decouple_mlps = nn.ModuleList([
            nn.Sequential(   
                nn.Linear(embedding_dim, embedding_dim//2),
                nn.Tanh(),
                nn.Linear(embedding_dim//2, self.out_size)
            ) for i in range(12)])

        #=========================== partial supervision ===========================
        att_num_list = [13, 4, 5181, 62, 21, 38, 16, 24, 7, 4, 5]  # 11个属性
        self.attr_classify = nn.ModuleList([
            nn.Sequential(   
                nn.Linear(self.out_size, self.out_size//2),
                nn.Tanh(),
                nn.Linear(self.out_size//2, att_num_list[i])
            ) for i in range(11)])
    
        #=========================== deconvolution ====================
        ngf = 128
        self.decoder = nn.Sequential(
                    nn.ConvTranspose2d( self.att_num*self.out_size, ngf * 8, 4, 1, 0,0, bias=False),  
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 3, 2,0, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 3, 2,2, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( ngf * 2, ngf, 4, 3, 2,0, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    nn.ConvTranspose2d( ngf, 3, 4, 3, 2, 2, bias=False),
                    nn.Tanh()
                )       
        self.mse = torch.nn.MSELoss(reduction='mean') 

        #=========================== attribute-level graph =========================
        self.attr_ego = nn.Sequential(
            nn.Linear(self.out_size, self.out_size),
            nn.LeakyReLU(),
        )
        self.attr_mess = nn.Sequential(
            nn.Linear(self.out_size, self.out_size),
            nn.LeakyReLU(),
        )
        self.attr_attention = nn.Sequential(
            nn.Linear(self.out_size * 2, self.out_size),
            nn.LeakyReLU(),
            nn.Linear(self.out_size, 1)
        )
        self.attr_comp_classify = nn.Sequential(
            nn.Linear(self.out_size, self.out_size // 2),
            nn.LeakyReLU(),
            nn.Linear(self.out_size // 2, 1)
        )

        #=========================== overview-level graph =========================
        self.overview_fc = nn.Linear(embedding_dim, self.out_size, bias=False)
        self.overview_att_fc = nn.Linear(self.out_size*12, self.out_size, bias=False)

        self.overview_ego = nn.Sequential(
            nn.Linear(self.out_size*2, self.out_size),
            nn.LeakyReLU()
        )
        self.overview_mess = nn.Sequential(
            nn.Linear(self.out_size*2, self.out_size),
            nn.LeakyReLU()
        ) 
        self.overview_attention = nn.Sequential(
            nn.Linear(self.out_size * 4, self.out_size),
            nn.LeakyReLU(),
            nn.Linear(self.out_size, 1)
        )
        # overview score
        self.overview_comp_classify = nn.Sequential(
            nn.Linear(self.out_size, self.out_size // 2),
            nn.LeakyReLU(),
            nn.Linear(self.out_size // 2, 1)
        )

        # ============== final classify ============
        self.classify = nn.Sequential(
            nn.Linear(13, 13),
            nn.LeakyReLU(),
            nn.Linear(13, 2)
        )  

    def extract_img_feature(self, img):
        batch_size = len(img)
        max_outfit = max([len(t) for t in img])
        y = torch.zeros((batch_size, max_outfit, 3, 224, 224))

        for index, t in enumerate(img):
            y[index][:min(len(t), self.outfit_threshold)] = torch.stack(t[:min(len(t), self.outfit_threshold)])
        y = y.cuda().permute(1,0,2,3,4).contiguous()
        feature = torch.zeros((max_outfit, batch_size, self.hidden_dim)).cuda()
        for index in range(max_outfit):
            feature[index] = self.backbone(y[index])
        feature = feature.permute(1,0,2).contiguous()    # max_out, batch_size, dim  -> batch_size, nodes, dim

        return feature                    # batch_size, nodes, dim   ;  batch_size, outfitnum, outfitnum

    def compute_ortho(self, x1, x2):
        x2 = x2.detach()
        ortho = torch.matmul(F.normalize(x1, dim=-1), F.normalize(x2.permute(0,2,1), dim=1))
        ortho = torch.diagonal(ortho, dim1=-2, dim2=-1)
        zeros = torch.zeros_like(ortho).cuda()     # batch_size, outfitnum
        loss_ortho = F.mse_loss(ortho, zeros)      

        return loss_ortho

    def partial_supervision(self, features, att_label, att_mask):  

        # att_mask (batch_size,item_num,att_num)  16,10,12,1
        # att_label (batch_size, item_num, att_num)  16,10,11
        batch_size = att_label.size()[0]
        o_num = att_label.size()[1]
        att_label = att_label.split(1,dim=2) 
        att_mask = att_mask > 0
        att_mask = att_mask[:,:,:-1,:].squeeze(-1).split(1, dim=2)
        
        batch_loss = 0
        for i in range(11):
            label = att_label[i].view(batch_size * o_num, -1).squeeze()
            predict = self.attr_classify[i](features[i]).view(batch_size * o_num, -1)
            mask = att_mask[i].view(batch_size * o_num).squeeze()
            batch_loss += F.cross_entropy(predict[mask], label[mask])

        return batch_loss

    def dc_img_feature(self, img_f, att_mask, att_label, img, partial_mask):
        #partial_mask : batch_size, outfit_num, 12, 1
        #img_f:  batch_size, nodes, dim 
        # att_mask: batch_size, outfit_num, 12, 1
        outfit_mask = torch.sum(att_mask, dim=-2)  # batch_size, 10, 1
        outfit_mask[outfit_mask > 0] = 1 

        list_b = []
        for l in img: #batch
            outfit_img = torch.stack(l,0)
            #print('+++++',outfit_img.size())   # 10,3,224,224
            list_b.append(outfit_img)
            pass
        img = torch.stack(list_b,0)
        img = img.cuda()
        batch_size, outfit_num, att_num, _ = att_mask.size()

        decouple_feature = [self.decouple_mlps[i](img_f) for i in range(12)]   # 12 feature list (12,batch_size,node,64)

        # partial supervision
        partial_supervision_loss = self.partial_supervision(decouple_feature[:-1], att_label, att_mask)

        # ortho regularization
        ortho_loss = 0
        for index, i in enumerate(decouple_feature[:-1]):
            ortho_loss += self.compute_ortho(decouple_feature[-1] * partial_mask[:,:,-1,:], i * partial_mask[:,:,index,:])
        
        for i in range(11):
            decouple_feature[i] = decouple_feature[i] * partial_mask[:,:,i,:]
        decouple_feature[11] = decouple_feature[11] * partial_mask[:,:,11,:]
        # deconvolution regularization
        outfit_img_final = torch.cat(decouple_feature, 2).view(batch_size * outfit_num, -1, 1, 1)
        outfit_mask = outfit_mask.view(batch_size * outfit_num, 1).unsqueeze(-1).unsqueeze(-1)
        img_d_x = self.decoder(outfit_img_final)
        img_origin_y = img.view(batch_size * outfit_num, 3, 224, 224)
        deconvolution_loss = self.mse(img_d_x * outfit_mask, img_origin_y * outfit_mask)

        return partial_supervision_loss, ortho_loss, deconvolution_loss, torch.stack(decouple_feature) # 12, batch_size, 10, 64


    def attribute_graph(self, decouple_feature, mask):
        attr_num, batch_size, outfit_num, dim = decouple_feature.size()

        out_feature = torch.zeros_like(decouple_feature).cuda()  # 12, batch_size, 10, 64
        for i in range(12):
            attr_feature = decouple_feature[i,:,:,:] # batch_size, 10, 64
            attr_mask = mask[:,:,i,:]   # batch_size, 10, 1
            adj = torch.ones((batch_size, outfit_num, outfit_num)).cuda()
            for j in range(batch_size):   # adj
                attr_mask_batch = attr_mask[j].squeeze(-1)  # 10,
                attr_mask_batch = attr_mask_batch < 1 
                adj[j,attr_mask_batch,:] = 0
                adj[j,:,attr_mask_batch] = 0

                diag = torch.diag(adj[j])   
                diag = torch.diag_embed(diag)  
                adj[j,:,:] = adj[j,:,:] - diag 

            # attr_feature: batch_size, 10, 64
            # adj         : batch_size, 10, 10
            atten_input = torch.cat([attr_feature.repeat(1,1,outfit_num).view(batch_size, outfit_num * outfit_num, dim), attr_feature.repeat(1,outfit_num,1)], dim=2).view(-1, outfit_num, outfit_num, 2*dim)
            e = self.attr_attention(atten_input).view(batch_size, outfit_num, outfit_num)  # batch_size, outfit_num, outfit_num
            zero_vec = -9e15 * torch.ones_like(e).cuda()
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=-1) # batch_size, outfit_num, outfit_num

            relation = attr_feature.repeat(1, 1, outfit_num).view(batch_size, outfit_num * outfit_num, dim) * attr_feature.repeat(1, outfit_num, 1) # batch_size, outfit_num * outfit_num, dim
            relation = relation.view(batch_size, outfit_num, outfit_num, dim) * adj.unsqueeze(-1)  # batch_size, outfit_num, outfit_num, dim
            relation = relation * attention.unsqueeze(-1)
            relation = torch.sum(relation, dim=2)  # batch_size, outfit_num, dim
 
            neighbor_info = self.attr_mess(relation)
            self_info = self.attr_ego(attr_feature)
            update_info = (self_info + neighbor_info) * attr_mask

            out_feature[i,:,:,:] = update_info 
        
        return out_feature
    
    def overview_graph(self, attr_graph_feature, img_feature, mask):   
        # attr_graph_feature: 12, batch_size, 10, 64
        # img_feature: batch_size, 10, 256   
        # mask: batch_size, 10, 12, 1
        attr_num, batch_size, outfit_num, dim = attr_graph_feature.size()
        img_feature = self.overview_fc(img_feature) # batch_size, 10, 64 
        attr_graph_feature = attr_graph_feature.permute(1,2,0,3).contiguous().view(batch_size, outfit_num,attr_num*dim) # batch_size, 10,12* 64  
        attr_graph_feature = self.overview_att_fc(attr_graph_feature)
        img_feature = torch.cat([img_feature,attr_graph_feature],dim= 2)

        mask_attr = mask.squeeze(-1) # batch_size, 10, 12   
        mask_per_outfit = torch.sum(mask, dim=2).squeeze(-1) # batch_size, 10  
        mask_per_outfit[mask_per_outfit > 0] = 1 

        outfit_adj = torch.ones((batch_size, outfit_num, outfit_num)).cuda()
        for j in range(batch_size):
            mask_per_batch = mask_per_outfit[j]
            mask_per_batch = mask_per_batch < 1
            outfit_adj[j,mask_per_batch,:] = 0   #padding set 0
            outfit_adj[j,:,mask_per_batch] = 0

            diag = torch.diag(outfit_adj[j])
            diag = torch.diag_embed(diag)
            outfit_adj[j,:,:] = outfit_adj[j,:,:] - diag  # adj


        adj_mask = torch.sum(outfit_adj, dim=-1)
        adj_mask[adj_mask > 0] = 1

        atten_input = torch.cat([img_feature.repeat(1,1,outfit_num).view(batch_size, outfit_num * outfit_num, dim*2), img_feature.repeat(1,outfit_num,1)], dim=2).view(-1, outfit_num, outfit_num, 4*dim)
        e = self.overview_attention(atten_input).view(batch_size, outfit_num, outfit_num)  # batch_size, outfit_num, outfit_num
        zero_vec = -9e15 * torch.ones_like(e).cuda()
        attention = torch.where(outfit_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1) # batch_size, outfit_num, outfit_num

        relation = img_feature.repeat(1, 1, outfit_num).view(batch_size, outfit_num * outfit_num, dim*2) * img_feature.repeat(1, outfit_num, 1) # batch_size, outfit_num * outfit_num, dim
        relation = relation.view(batch_size, outfit_num, outfit_num, dim*2) * outfit_adj.unsqueeze(-1)  # batch_size, outfit_num, outfit_num, dim
        relation = relation * attention.unsqueeze(-1)
        relation = torch.sum(relation, dim=2)  # batch_size, outfit_num, dim   

        neighbor_info = self.overview_mess(relation)
        self_info = self.overview_ego(img_feature)
        update_info = (self_info + neighbor_info)       # batch_size, 130, dim
        update_info = adj_mask.unsqueeze(-1) * update_info
        return update_info

 
    def compute_attr_compatibility_score(self, graph_feature, att_mask):
        # graph_feature: 12, batch_size, outfit_num, dim
        # att_mask: batch_size, outfit_num, 12, 1
        batch_size, _,_,_ = att_mask.size()
        compatibility_score = torch.zeros((batch_size, 12))
        score = self.attr_comp_classify(graph_feature).permute(1,2,0,3) # 12, batch_size, outfit_num, 1 -> batch_size, outfit_num, 12, 1   
        score = torch.sum(att_mask * score, dim=1) # batch_size, 12, 1
        norm = torch.sum(att_mask, dim=1)
        norm[norm == 0] = 1 # 0/0  -> 0/1
        score = torch.div(score, norm) # batch_size, 12, 1

        return score.squeeze(-1)
    
    def compute_overview_compatibility_score(self, graph_feature, partial_mask):
        # graph_feature: 16,10,64
        # partial_mask: 16, 10, 12, 1
        
        batch_size, _,_,_ = partial_mask.size()
        score = self.overview_comp_classify(graph_feature) # batch_size, 10, 1 

        norm = torch.sum(partial_mask, dim=-2)  # batch_size, 10, 1
        norm[norm > 0] = 1 
        score = torch.sum(score * norm, dim=1)
        norm = torch.sum(norm, dim=1) # batch_size, 1
        norm[norm == 0] = 1 
        score = torch.div(score, norm)
        return score
        
    
    def outfit_compatibility_score(self, feature):
        return self.classify(feature)


    def forward(self, img_origin, att_mask, att_label, partial_mask):
        img = copy.deepcopy(img_origin)
        img_feature = self.extract_img_feature(img)    

        partial_supervision_loss, ortho_loss, deconvolution_loss, decouple_features = self.dc_img_feature(img_feature, att_mask, att_label, img, partial_mask)
        attr_graph_feature = self.attribute_graph(decouple_features, partial_mask)
        attr_score = self.compute_attr_compatibility_score(attr_graph_feature, partial_mask)

        overview_graph_feature = self.overview_graph(attr_graph_feature, img_feature, partial_mask)
        overview_score = self.compute_overview_compatibility_score(overview_graph_feature, partial_mask)

        combine_score = torch.cat([overview_score, attr_score], dim=-1)
        score = self.outfit_compatibility_score(combine_score)

        return score, partial_supervision_loss, deconvolution_loss, ortho_loss