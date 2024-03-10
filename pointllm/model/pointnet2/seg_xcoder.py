from torch import nn
import torch
from .pointnet2_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg, PointNetFeaturePropagation
# from knn_cuda import KNN
# from timm.models.layers import DropPath, trunc_normal_
from .transformer import TwoWayTransformer

from torch.nn import functional as F



class PointEncoder(nn.Module):
    def __init__(self, config,normal_channel=True):
        super().__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        if config.npoints==8192:
            npoint_list = [2048,512]
        else:
            npoint_list = [1024,512]

        self.sa1 = PointNetSetAbstractionMsg(npoint_list[0], [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint_list[1], [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256+6+additional_channel, mlp=[256, 256])
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self,xyz):
        xyz = xyz.transpose(1,2).contiguous()
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)

        return (l2_points,l1_points,l0_points),(l2_xyz,l1_xyz,l0_xyz)

class PromptEncoder(nn.Module):
    def __init__(self, config,text_feature_dim = 512,out_dim = 256):
        super().__init__()

        self.embed_dim = out_dim

        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

        self.prompt_linear = nn.Linear(text_feature_dim, self.embed_dim)

    def _get_batch_size(self,prompt):
        if prompt is not None:
            return prompt.shape[0]
        else:
            return 1

    def forward(self, prompt):
        prompt = self.prompt_linear(prompt)
        bs = self._get_batch_size(prompt)
        sparse_embeddings = torch.empty((bs,0,self.embed_dim), device = self.no_mask_embed.weight.device)

        if prompt is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, prompt], dim = 1)

        # dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        #     bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        # )
        
        return sparse_embeddings




class MaskDecoder(nn.Module):
    def __init__(self, config,out_dim=256,iou_head_depth=3,iou_head_hidden_dim=256, **kwargs):
        super().__init__()

        self.trans_dim = out_dim

        self.reduce_dim = nn.Linear(self.trans_dim, self.trans_dim//8)

        self.num_mask_tokens = 3 + 1

        self.transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            )

        self.iou_token = nn.Embedding(1, self.trans_dim)
        self.mask_token = nn.Embedding(self.num_mask_tokens, self.trans_dim)


        self.iou_prediction_head = MLP(
            self.trans_dim, iou_head_hidden_dim, 1, iou_head_depth
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(self.trans_dim, self.trans_dim, self.trans_dim//8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )


        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  



    def forward(self,
                pts,
                point_embeddings,
                sparse_prompt_embeddings,
                ):
        
        # point_pe = self.pos_embed(pts)
        masks, iou_pred = self.predict_masks(pts,point_embeddings, sparse_prompt_embeddings)

        mask_slice = slice(0,1)
        masks = masks[:,mask_slice,:]
        iou_pred = iou_pred[:,mask_slice]

        return masks, iou_pred
        
    def predict_masks(self,pts,point_embeddings, sparse_prompt_embeddings):
        
        output_tokens = torch.cat([self.iou_token.weight, self.mask_token.weight], dim = 0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.shape[0], -1, -1
        )

        tokens = torch.cat([output_tokens,sparse_prompt_embeddings], dim = 1)

        for i in range(len(pts)):

            src  = point_embeddings[i].transpose(-1,-2)
            # print(src.shape)
            
            pos_src = self.pos_embed(pts[i].transpose(-1,-2))

            hs, src = self.transformer(src,pos_src, tokens)
            iou_token_out = hs[:,0,:]
            mask_tokens_out = hs[:,1:1+self.num_mask_tokens,:]  # num_mask_tokens is 1

            # upscale
            src = src.contiguous()
            # print(src.shape)
            # upscaled_embedding = self.output_upscaling(src,pts,ori_pts)


        src = self.reduce_dim(src)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # print(upscaled_embedding.shape)
        masks = (hyper_in @ src.transpose(-1,-2))

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks,iou_pred



class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class Config():
    def __init__(self):
        self.npoints = 8192

if __name__ == "__main__":
    config = Config()
    encoder = PointEncoder(config)

    x = torch.randn(10,6,8192)
    a,b = encoder(x)

    print(a)

