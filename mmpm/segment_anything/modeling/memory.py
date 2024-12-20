# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from model.group_modules import *
# from model import resnet
# from model.cbam import CBAM

# class FeatureFusionBlock(nn.Module):
#     def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
#         super().__init__()

#         self.distributor = MainToGroupDistributor()
#         self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
#         self.attention = CBAM(g_mid_dim)
#         self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

#     def forward(self, x, g):
#         batch_size, num_objects = g.shape[:2]

#         g = self.distributor(x, g)
#         g = self.block1(g)
#         r = self.attention(g.flatten(start_dim=0, end_dim=1))
#         r = r.view(batch_size, num_objects, *r.shape[1:])

#         g = self.block2(g+r)

#         return g