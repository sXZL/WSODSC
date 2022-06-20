# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import torch
from torch import nn
import torch.nn.functional as F

from wetectron.modeling import registry


@registry.ROI_WEAK_PREDICTOR.register("WSDDNPredictor")
class WSDDNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(WSDDNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2
        
        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x)
        
        if not self.training:    
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)    
        else:
            final_det_logit = det_logit
            
        return cls_logit, final_det_logit, None
        

@registry.ROI_WEAK_PREDICTOR.register("OICRPredictor")
class OICRPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(OICRPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)
        
        print (self.cls_score)
    
        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.ref3 = nn.Linear(num_inputs, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2
        
        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x) 
        ref1_logit = self.ref1(x)
        ref2_logit = self.ref2(x)        
        ref3_logit = self.ref3(x)
        
        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
            # 
            ref1_logit = F.softmax(ref1_logit, dim=1)
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
        else:
            final_det_logit = det_logit
            
        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        return cls_logit, final_det_logit, ref_logits


@registry.ROI_WEAK_PREDICTOR.register("MISTPredictor")
class MISTPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MISTPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)
    
        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred1 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        
        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred2 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        
        self.ref3 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred3 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2
        
        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x) 
        ref1_logit = self.ref1(x)
        bbox_pred1 = self.bbox_pred1(x)
        ref2_logit = self.ref2(x)
        bbox_pred2 = self.bbox_pred2(x)     
        ref3_logit = self.ref3(x)
        bbox_pred3 = self.bbox_pred3(x)
        
        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
            ref1_logit = F.softmax(ref1_logit, dim=1)
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
        else:
            final_det_logit = det_logit
            
        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        bbox_preds = [bbox_pred1, bbox_pred2, bbox_pred3]
        return cls_logit, final_det_logit, ref_logits, bbox_preds


@registry.ROI_WEAK_PREDICTOR.register("OICRPredictorCTX")
class OICRPredictorCTX(nn.Module):
    def __init__(self, config, in_channels):
        super(OICRPredictorCTX, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)
        self.ctx_score = nn.ModuleList([nn.Linear(num_inputs, num_classes) for _ in range(config.CTX.NUM_BRANCHES)])

    
        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.ref3 = nn.Linear(num_inputs, num_classes)

        self._initialize_weights()

        self.comb_ctx_method = config.CTX.COMB_CTX_METHOD
        self.use_ctx = config = config.CTX.USE_CTX

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, ctx, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2
        
        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x) 
        ctx_logit = self.forward_ctx_boxes(ctx, cls_logit.shape[0]) 
        ref1_logit = self.ref1(x)
        ref2_logit = self.ref2(x)        
        ref3_logit = self.ref3(x)

        if self.use_ctx == 'add':
            det_logit = det_logit + ctx_logit
        elif self.use_ctx == 'sub':
            det_logit = det_logit - ctx_logit
        elif self.use_ctx == 'mul':
            det_logit = det_logit * ctx_logit
        else:
            NotImplementedError
        
        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
            # 
            ref1_logit = F.softmax(ref1_logit, dim=1)
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
        else:
            final_det_logit = det_logit
            
        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        return cls_logit, final_det_logit, ref_logits
    
    def forward_ctx_boxes(self, ctx, n_proposals):
        ctx = ctx.reshape(n_proposals, -1, 4096) # [2502, 4, 4096]
        ctx_logits = [] # [(1502, 20)]
        for i in range(ctx.shape[1]):
            ctx_branch_i = i % len(self.ctx_score) # when their is only one ctx branch, always use that branch
            ctx_logit = self.ctx_score[ctx_branch_i](ctx[:, i ,:])
            ctx_logits.append(ctx_logit)

        if self.comb_ctx_method == 'mean':
            return torch.stack(ctx_logits).mean(dim=0)
        elif self.comb_ctx_method == 'max':
            return torch.stack(ctx_logits).max(dim=0)[0]
        elif self.comb_ctx_method == 'min':
            return torch.stack(ctx_logits).min(dim=0)[0]
        else:
            NotImplementedError


@registry.ROI_WEAK_PREDICTOR.register("MISTPredictorCTX")
class MISTPredictorCTX(nn.Module):
    def __init__(self, config, in_channels):
        super(MISTPredictorCTX, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        self.det_score = nn.Linear(num_inputs, num_classes)
        self.ctx_score = nn.ModuleList([nn.Linear(num_inputs, num_classes) for _ in range(config.CTX.NUM_BRANCHES)])
    
        self.ref1 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred1 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        
        self.ref2 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred2 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        
        self.ref3 = nn.Linear(num_inputs, num_classes)
        self.bbox_pred3 = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        self._initialize_weights()

        self.comb_ctx_method = config.CTX.COMB_CTX_METHOD
        self.use_ctx = config = config.CTX.USE_CTX

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, ctx, proposals):
        if x.dim() == 4:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        assert x.dim() == 2
        
        cls_logit = self.cls_score(x)
        det_logit = self.det_score(x) 
        ctx_logit = self.forward_ctx_boxes(ctx, cls_logit.shape[0]) 
        ref1_logit = self.ref1(x)
        bbox_pred1 = self.bbox_pred1(x)
        ref2_logit = self.ref2(x)
        bbox_pred2 = self.bbox_pred2(x)     
        ref3_logit = self.ref3(x)
        bbox_pred3 = self.bbox_pred3(x)

        if self.use_ctx == 'add':
            det_logit = det_logit + ctx_logit
        elif self.use_ctx == 'sub':
            det_logit = det_logit - ctx_logit
        elif self.use_ctx == 'mul':
            det_logit = det_logit * ctx_logit
        else:
            NotImplementedError
        
        if not self.training:
            cls_logit = F.softmax(cls_logit, dim=1)
            # do softmax along ROI for different imgs
            det_logit_list = det_logit.split([len(p) for p in proposals])
            final_det_logit = []
            for det_logit_per_image in det_logit_list:
                det_logit_per_image = F.softmax(det_logit_per_image, dim=0)
                final_det_logit.append(det_logit_per_image)
            final_det_logit = torch.cat(final_det_logit, dim=0)
            ref1_logit = F.softmax(ref1_logit, dim=1)
            ref2_logit = F.softmax(ref2_logit, dim=1)
            ref3_logit = F.softmax(ref3_logit, dim=1)
        else:
            final_det_logit = det_logit
            
        ref_logits = [ref1_logit, ref2_logit, ref3_logit]
        bbox_preds = [bbox_pred1, bbox_pred2, bbox_pred3]
        return cls_logit, final_det_logit, ref_logits, bbox_preds
    
    def forward_ctx_boxes(self, ctx, n_proposals):
        ctx = ctx.reshape(n_proposals, -1, 4096) # [2502, 4, 4096]
        ctx_logits = [] # [(1502, 20)]
        for i in range(ctx.shape[1]):
            ctx_branch_i = i % len(self.ctx_score) # when their is only one ctx branch, always use that branch
            ctx_logit = self.ctx_score[ctx_branch_i](ctx[:, i ,:])
            ctx_logits.append(ctx_logit)

        if self.comb_ctx_method == 'mean':
            return torch.stack(ctx_logits).mean(dim=0)
        elif self.comb_ctx_method == 'max':
            return torch.stack(ctx_logits).max(dim=0)[0]
        elif self.comb_ctx_method == 'min':
            return torch.stack(ctx_logits).min(dim=0)[0]
        else:
            NotImplementedError

def make_roi_weak_predictor(cfg, in_channels):
    func = registry.ROI_WEAK_PREDICTOR[cfg.MODEL.ROI_WEAK_HEAD.PREDICTOR]
    return func(cfg, in_channels)
