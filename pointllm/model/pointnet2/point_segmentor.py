from .seg_xcoder import PointEncoder,PromptEncoder,MaskDecoder
import torch
import torch.nn as nn

class PointSegTransformer(nn.Module):
    def __init__(self,config,use_max_pool=False,**kwargs):
        super().__init__()

        self.config = config

        self.mask_threshold = getattr(self.config.model,'mask_threshold',0.0)

        self.point_encoder = PointEncoder(self.config.model)
        self.prompt_encoder = PromptEncoder(self.config.model)
        self.mask_decoder = MaskDecoder(self.config.model)

    def forward(self,pts,prompts):
        point_embeddings,pts = self.point_encoder(pts)
        sparse_prompt_embeddings = self.prompt_encoder(prompts)
        masks, iou_pred = self.mask_decoder(pts,point_embeddings,sparse_prompt_embeddings)

        return masks[:,0,:], iou_pred
    
    def inference(self,pts,prompts):
        point_embeddings,pts = self.point_encoder(pts)
        sparse_prompt_embeddings = self.prompt_encoder(prompts)
        masks, iou_pred = self.mask_decoder(pts,point_embeddings,sparse_prompt_embeddings)

        mask = mask>self.mask_threshold

        return masks[:,0,:], iou_pred
    
class Config():
    def __init__(self):
        self.model = None
        self.npoint = 8192
        # self.config = None

if __name__ == "__main__":
    
    config = Config()
    config.model = Config()
    config.model.npoints = 8192
    segmentor = PointSegTransformer(config).cuda()

    pts = torch.rand(10,6,8192).cuda()
    prompts = torch.rand(10,6,512).cuda()

    mask,iou = segmentor(pts,prompts)

    print(mask.shape)
    print(iou.shape)
