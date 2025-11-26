import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torchvision.models as tvmodels
import math

# -------------------------
# Image encoder: returns region embeddings (B, m, dv) then projects to (B, m, dt)
# -------------------------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=768, freeze_backbone=True, input_size=112):
        super().__init__()
        # use torchvision ResNet50 and keep conv layers (no avgpool / fc)
        # resnet = tvmodels.resnet50(pretrained=True)
        resnet = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.DEFAULT)

        # remove avgpool and fc
        self.backbone_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        if freeze_backbone:
            for p in self.backbone_conv.parameters():
                p.requires_grad = False

        # dv = 2048 for resnet50 last conv output
        self.dv = 2048
        # project dv -> embed_dim with 1x1 conv (keeps spatial regions)
        self.proj_conv = nn.Conv2d(self.dv, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim
        self.input_size = input_size

    def forward(self, x):
        # x: (B, C=3, H, W) expect resized to input_size (e.g.,112)
        feat_map = self.backbone_conv(x)            # (B, 2048, H', W')  (for 112x112 -> 4x4)
        # project
        proj = self.proj_conv(feat_map)             # (B, dt, H', W')
        B, dt, Hf, Wf = proj.shape
        m = Hf * Wf
        # flatten spatial dims -> region embeddings
        regions = proj.view(B, dt, m).permute(0, 2, 1)  # (B, m, dt)
        return regions   # (B, m, dt)

# -------------------------
# Text encoder: returns token embeddings (B, n, dt) and cls (B, dt)
# -------------------------
class TextEncoder(nn.Module):
    def __init__(self, bert_name="bert-base-chinese", fine_tune=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False
        # hidden size autodetected from model config
        self.dt = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        # returns:
        #   token_embeds: (B, n, dt)
        #   cls: (B, dt)
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        token_embeds = out.last_hidden_state      # (B, n, dt)
        cls = token_embeds[:, 0, :]               # (B, dt)  (CLS token)
        return token_embeds, cls

# -------------------------
# Multi-modal masked attention module (implements masked attention described)
# -------------------------
class MultiModalMaskedAttention(nn.Module):
    def __init__(self, dt, hidden_ffn=512):
        super().__init__()
        self.dt = dt
        # linear projections for Q,K,V (we'll have two directions; reuse modules with appropriate inputs)
        self.Wq_v = nn.Linear(dt, dt)   # project image regions -> Q
        self.Wk_t = nn.Linear(dt, dt)   # project text tokens -> K
        self.Wv_t = nn.Linear(dt, dt)   # project text tokens -> V

        self.Wq_t = nn.Linear(dt, dt)   # project text tokens -> Q (for opposite direction)
        self.Wk_v = nn.Linear(dt, dt)   # project image regions -> K
        self.Wv_v = nn.Linear(dt, dt)   # project image regions -> V

        # FFN to predict threshold l per-sample
        self.threshold_ffn = nn.Sequential(
            nn.Linear(dt * 2, hidden_ffn),
            nn.ReLU(),
            nn.Linear(hidden_ffn, 1),
            nn.Sigmoid()   # produce value in (0,1) to be used as threshold
        )

    def forward(self, oV, oT, t_mask=None):
        # oV: (B, m, dt)  image region embeddings
        # oT: (B, n, dt)  text token embeddings
        B, m, dt = oV.shape
        _, n, _ = oT.shape
        device = oV.device

        # compute similarity matrix S = softmax( (oV @ oT^T) / sqrt(d) )  along text dim maybe
        sim_raw = torch.matmul(oV, oT.transpose(1, 2))     # (B, m, n)
        # compute per-sample threshold l from pooled features (mean pooling)
        v_pool = oV.mean(dim=1)  # (B, dt)
        t_pool = oT.mean(dim=1)  # (B, dt)
        l = self.threshold_ffn(torch.cat([v_pool, t_pool], dim=-1)).squeeze(-1)  # (B,) in (0,1)

        # build mask M where S >= l -> but S not yet normalized; use normalized similarity
        # We'll compute scaled scores and then set masked positions to -inf before softmax.
        scaled = sim_raw / math.sqrt(dt)   # (B,m,n)

        # create per-sample threshold tensor to compare with raw similarity after a sigmoid normalization:
        # To emulate "S = softmax(...) then M_{i,j} = 1 if S_ij >= l", we need S normalized along text dim.
        S = F.softmax(scaled, dim=-1)     # (B,m,n)
        # l: (B,) -> (B,1,1)
        l_exp = l.view(B, 1, 1)
        M = (S >= l_exp).to(dtype=torch.bool)   # boolean mask (B,m,n)

        # ---- image (Q) <- text (K,V) masked attention ----
        Q_v = self.Wq_v(oV)       # (B,m,dt)
        K_t = self.Wk_t(oT)       # (B,n,dt)
        V_t = self.Wv_t(oT)       # (B,n,dt)

        scores_vt = torch.matmul(Q_v, K_t.transpose(1,2)) / math.sqrt(dt)  # (B,m,n)
        # mask positions where M is False
        neg_inf = -1e9
        scores_vt_masked = scores_vt.masked_fill(~M, neg_inf)
        attn_vt = F.softmax(scores_vt_masked, dim=-1)   # (B,m,n)
        out_vt = torch.matmul(attn_vt, V_t)             # (B,m,dt)

        # ---- text (Q) <- image (K,V) masked attention ----
        Q_t = self.Wq_t(oT)       # (B,n,dt)
        K_v = self.Wk_v(oV)       # (B,m,dt)
        V_v = self.Wv_v(oV)       # (B,m,dt)

        # for opposite direction, need mask shape (B,n,m): basically transpose M
        M_tv = M.transpose(1,2)   # (B,n,m)
        scores_tv = torch.matmul(Q_t, K_v.transpose(1,2)) / math.sqrt(dt)  # (B,n,m)
        scores_tv_masked = scores_tv.masked_fill(~M_tv, neg_inf)
        attn_tv = F.softmax(scores_tv_masked, dim=-1)  # (B,n,m)
        out_tv = torch.matmul(attn_tv, V_v)            # (B,n,dt)

        # we return attended outputs (region-level and token-level)
        # out_vt: (B,m,dt)  image regions enhanced by text
        # out_tv: (B,n,dt)  text tokens enhanced by image
        return out_vt, out_tv, M.float()

# -------------------------
# Fusion head: pool and classify  (fixed)
# -------------------------
class FusionHead(nn.Module):
    def __init__(self, dim=768, hidden=768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim*2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, t_cls, t_attended_tokens=None, v_attended_regions=None):
        # t_cls: (B, dt)
        # t_attended_tokens: (B, n, dt) or None
        # v_attended_regions: (B, m, dt) or None
        if t_attended_tokens is not None:
            t_pool = t_attended_tokens.mean(dim=1)
        else:
            t_pool = t_cls

        if v_attended_regions is not None:
            v_pool = v_attended_regions.mean(dim=1)
        else:
            # اگر تصویر موجود نیست، یک بردار صفر بساز (هم‌شکل t_pool)
            v_pool = torch.zeros_like(t_pool)

        x = torch.cat([t_pool, v_pool], dim=-1)   # (B, 2*dt)
        h = self.gate(x)
        logit = self.classifier(h).squeeze(-1)
        return logit, h  # برمی‌گردانیم هم لاگیت هم بردار فیوز‌شده h (دلخواه)


# -------------------------
# Full model (forward تغییر یافته)
# -------------------------
class CMMTN_PU(nn.Module):
    def __init__(self, bert_name="bert-base-chinese", freeze_cnn=True, ft_bert=True,
                 embed_dim=768, text_proj_dim=None):
        super().__init__()
        # Text encoder
        self.text = TextEncoder(bert_name=bert_name, fine_tune=ft_bert)
        self.text_proj = nn.Linear(embed_dim, text_proj_dim) if text_proj_dim is not None else None

        # Vision encoder
        self.vision = ImageEncoder(embed_dim=embed_dim, freeze_backbone=freeze_cnn)

        # Multi-modal masked attention
        self.mm_attn = MultiModalMaskedAttention(dt=embed_dim)


        # Fusion & classification head
        self.fusion = FusionHead(dim=embed_dim, hidden=embed_dim)

    def forward(self, input_ids, attention_mask, image, return_mask=False):
        # -------------------
        # Text embeddings
        # -------------------
        # token_embeds: (B, L, D), cls: (B, D)
        token_embeds, cls = self.text(input_ids, attention_mask)
        if self.text_proj is not None:
            token_embeds = self.text_proj(token_embeds)
            cls = self.text_proj(cls)

        # -------------------
        # Vision embeddings
        # -------------------
        # regions: (B, M, D)
        regions = self.vision(image)

        # -------------------
        # Multi-modal masked attention
        # -------------------
        # out_vt: visual-to-text, out_tv: text-to-visual, mask: attention mask
        out_vt, out_tv, mask = self.mm_attn(regions, token_embeds)

        # -------------------
        # Fusion & classification
        # -------------------
        logit, fused_vec = self.fusion(cls, out_tv, out_vt)

        if return_mask:
            return logit, mask
        return logit

