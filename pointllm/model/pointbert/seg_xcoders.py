from torch import nn
import torch
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from timm.models.layers import DropPath, trunc_normal_
from .transformer import TwoWayTransformer

from torch.nn import functional as F



class PointEncoder(nn.Module):
    def __init__(self, config, out_dim = 256, use_max_pool = False):
        super().__init__()

        self.config = config

        self.use_max_pool = use_max_pool

        self.trans_dim = config.trans_dim
        # self.trans_dim = out_chans
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.point_dims = config.point_dims
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        self.encoder = Encoder(point_input_dims=config.point_dims,encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.out = nn.Linear(self.trans_dim, out_dim)

        self.pts = None
        # self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        # self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        # self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3 + 16, mlp = [self.trans_dim * 4, self.trans_dim])
        # self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        # self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)

    # def get_pts(self):
    #     return self.pts

    def forward(self, pts):
        B,N,C = pts.shape
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        self.pts = center
        # # generate mask
        # bool_masked_pos = self._mask_center(center, no_mask = False) # B G
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        self.pe = pos
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x) # * B, G + 1(cls token)(513), C(384)
        if not self.use_max_pool:
            return self.out(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1) # * concat the cls token and max pool the features of different tokens, make it B, 1, C
        return self.out(concat_f) # * B, 1, C(384 + 384)

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

        # self.output_upscaling = []

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        # self.propagation_2 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        # self.propagation_1= PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= self.trans_dim + 3, mlp = [self.trans_dim * 4, self.trans_dim,self.trans_dim//8])
        # self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        # self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)


    def forward(self,
                pts,
                ori_pts,
                point_embeddings,
                sparse_prompt_embeddings,
                ):
        
        point_pe = self.pos_embed(pts)
        masks, iou_pred = self.predict_masks(pts,ori_pts,point_embeddings, point_pe, sparse_prompt_embeddings)

        mask_slice = slice(0,1)
        masks = masks[:,mask_slice,:]
        iou_pred = iou_pred[:,mask_slice]

        return masks, iou_pred
        
    def predict_masks(self,pts,ori_pts,point_embeddings, point_pe, sparse_prompt_embeddings):
        
        output_tokens = torch.cat([self.iou_token.weight, self.mask_token.weight], dim = 0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.shape[0], -1, -1
        )

        tokens = torch.cat([output_tokens,sparse_prompt_embeddings], dim = 1)

        # src = torch.repeat_interleave(point_embeddings,tokens.shape[0], dim = 0)
        src  = point_embeddings
        # print(src.shape)
        
        # pos_src = torch.repeat_interleave(point_pe,tokens.shape[0], dim = 0)
        pos_src = point_pe

        b,c,n = src.shape

        hs, src = self.transformer(src,pos_src, tokens)
        iou_token_out = hs[:,0,:]
        mask_tokens_out = hs[:,1:1+self.num_mask_tokens,:]  # num_mask_tokens is 1

        # upscale
        src = src.contiguous()
        # print(src.shape)
        upscaled_embedding = self.output_upscaling(src,pts,ori_pts)


        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b,c,n = upscaled_embedding.shape
        # print(upscaled_embedding.shape)
        masks = (hyper_in @ upscaled_embedding.view(b, c, n)).view(b, -1, n)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks,iou_pred



    def output_upscaling(self,src,pts,ori_pts):
        center_level_0 = ori_pts.transpose(-1, -2).contiguous() 
        center_level_1 = pts.transpose(-1,-2).contiguous()

        # center_level_1 = fps(ori_pts, 512).transpose(-1, -2).contiguous()    # 16,3,512    
        # f_level_1 = center_level_1
        # center_level_2 = fps(pts, 256).transpose(-1, -2).contiguous()    # 16,3,256       
        # f_level_2 = center_level_2
        # center_level_3 = center.transpose(-1, -2).contiguous()           # 16,3,512     

        # # init the feature by 3nn propagation
        # f_level_3 = feature_list[2]  # 16,384,512
        # f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1]) # 16,384,256
        # f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0]) # 16,384,512

        # # bottom up
        # f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        # f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        # print(center_level_0.shape)
        # print(center_level_1.shape)
        # print(src.shape)
        f_level_0 =  self.propagation_0(center_level_0, center_level_1, points1=center_level_0, points2 = src.transpose(-1,-2).contiguous())

        return f_level_0
    


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



def fps(data, number):
    '''
        data B N 3
        number int
    '''
    B,N,C = data.shape
    if C>3:
        xyz = data[:,:,:3].contiguous()
    else:
        xyz = data
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:,:,:3].contiguous()
            rgb = data[:, :, 3:]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3

        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        # idx = self.knn(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz 
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, point_input_dims,encoder_channel):
        super().__init__()
        self.point_input_dims = point_input_dims
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , c= point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    # def forward(self, x, pos):
    #     feature_list = []
    #     fetch_idx = [3, 7, 11]
    #     for i, block in enumerate(self.blocks):
    #         x = block(x + pos)
    #         if i in fetch_idx:
    #             feature_list.append(x)
    #     return feature_list
    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x

from .pointnet2_utils import PointNetFeaturePropagation
class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 384, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 384),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q