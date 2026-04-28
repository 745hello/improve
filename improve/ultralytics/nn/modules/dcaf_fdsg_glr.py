# Ultralytics 🚀 AGPL-3.0
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect

GLR_WARMUP_ITERATIONS = 200
GLR_REWEIGHT_POWER = 0.5
DCAF_REDUCTION_RATIO = 8
FDSG_GATE_CONTENT_WEIGHT = 0.45
FDSG_GATE_SPATIAL_WEIGHT = 0.35
FDSG_GATE_PRIOR_WEIGHT = 0.20
GATE_MIN_VALUE = 0.05
GATE_MAX_VALUE = 0.95
NORMALIZATION_MIN_MEAN = 0.05


# ----------------------------
# Basic blocks
# ----------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Conv-BN-SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise + pointwise"""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1)
        self.pw = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


# ----------------------------
# Innovation-1: DCAF
# ----------------------------
class DetailEnhance(nn.Module):
    """Local contrast / edge-like enhancement (lightweight)."""
    def __init__(self, c):
        super().__init__()
        self.dw = DWConv(c, c, 3, 1)
        self.avg = nn.AvgPool2d(3, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # high-frequency proxy
        hf = x - self.avg(x)
        return self.dw(x + self.alpha * hf)


class SemanticAlign(nn.Module):
    """
    Lightweight semantic alignment:
    offset-like branch approximated by dynamic modulation (avoid heavy deform conv dependency).
    """
    def __init__(self, c):
        super().__init__()
        self.pre = Conv(c, c, 1, 1)
        self.mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(c // 4, c, 1, 1, 0),
            nn.Sigmoid()
        )
        self.post = DWConv(c, c, 3, 1)

    def forward(self, x):
        y = self.pre(x)
        g = self.mod(y)
        y = y * g
        y = self.post(y)
        return y


class MIBlendGateLite(nn.Module):
    """
    Lite gate with bottleneck to control params:
    cat(sem,det,id) -> 1x1 reduce -> 1x1 to 3 weights
    """
    def __init__(self, c, r=8):
        super().__init__()
        cr = max(16, c // r)
        self.reduce = nn.Sequential(
            nn.Conv2d(c * 3, cr, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True),
        )
        self.w = nn.Conv2d(cr, 3, 1, 1, 0, bias=True)

    def forward(self, f_sem, f_det, f_id):
        z = torch.cat([f_sem, f_det, f_id], dim=1)
        z = self.reduce(z)
        w = torch.softmax(self.w(z), dim=1)
        return w[:, 0:1], w[:, 1:2], w[:, 2:3]


class SemanticAlignLite(nn.Module):
    """Bottlenecked semantic align to avoid huge 1024->256->1024 fully."""
    def __init__(self, c, r=8):
        super().__init__()
        cr = max(16, c // r)
        self.pre = nn.Sequential(
            nn.Conv2d(c, cr, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True),
        )
        self.mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cr, max(8, cr // 4), 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, cr // 4), cr, 1, 1, 0),
            nn.Sigmoid()
        )
        self.post = nn.Sequential(
            # depthwise at reduced channels
            nn.Conv2d(cr, cr, 3, 1, 1, groups=cr, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True),
            nn.Conv2d(cr, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        y = self.pre(x)
        y = y * self.mod(y)
        return self.post(y)


class DetailEnhanceLite(nn.Module):
    """Reduced compute detail enhance; for high-level (P5) detail path is not critical."""
    def __init__(self, c):
        super().__init__()
        self.avg = nn.AvgPool2d(3, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.dw = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        hf = x - self.avg(x)
        return self.dw(x + self.alpha * hf)

class DCAF(nn.Module):
    def __init__(self, c_out=256, c_low: int | None = None, c_cur: int | None = None, c_high: int | None = None):
        super().__init__()
        self.c_out = c_out
        self._built = False

        # 占位，首次 forward 再按真实通道构建
        self.align_det = None
        self.align_sem = None
        self.align_cur = None
        self.refine_detail = None
        self.refine_semantic = None
        self.refine_current = None
        self.mix_gate = None
        self.fuse = None
        self.branch_logits = nn.Parameter(torch.zeros(3))
        self.residual_alpha = nn.Parameter(torch.tensor(0.0))

        # 若 parse_model 已提供通道信息，则在构造时直接建参，确保参数被优化器捕获
        if c_low is not None and c_cur is not None and c_high is not None:
            self._build(c_low, c_cur, c_high)

    def _build(self, c_low, c_cur, c_high, device=None, dtype=None):
        # 低层 -> c_out
        self.align_det = nn.Sequential(
            nn.Conv2d(c_low, self.c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
        )
        # 高层 -> c_out
        self.align_sem = nn.Sequential(
            nn.Conv2d(c_high, self.c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
        )
        # 当前层 -> c_out
        self.align_cur = nn.Sequential(
            nn.Conv2d(c_cur, self.c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
        )

        # 分支增强与动态门控
        self.refine_detail = DetailEnhanceLite(self.c_out)
        self.refine_semantic = SemanticAlignLite(self.c_out, r=DCAF_REDUCTION_RATIO)
        self.refine_current = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_out, 3, 1, 1, groups=self.c_out, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_out, self.c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
        )
        self.mix_gate = MIBlendGateLite(self.c_out, r=DCAF_REDUCTION_RATIO)

        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(self.c_out * 3, self.c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c_out, self.c_out, 3, 1, 1, groups=self.c_out, bias=False),
            nn.BatchNorm2d(self.c_out),
            nn.SiLU(inplace=True),
        )

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)
        self._built = True

    def forward(self, f_low, f_cur, f_high):
        # 尺寸对齐到 cur
        if f_low.shape[-2:] != f_cur.shape[-2:]:
            f_low = F.interpolate(f_low, size=f_cur.shape[-2:], mode="bilinear", align_corners=False)
        if f_high.shape[-2:] != f_cur.shape[-2:]:
            f_high = F.interpolate(f_high, size=f_cur.shape[-2:], mode="bilinear", align_corners=False)

        if not self._built:
            self._build(
                c_low=f_low.shape[1],
                c_cur=f_cur.shape[1],
                c_high=f_high.shape[1],
                device=f_cur.device,
                dtype=f_cur.dtype,
            )

        d = self.refine_detail(self.align_det(f_low))
        c = self.refine_current(self.align_cur(f_cur))
        s = self.refine_semantic(self.align_sem(f_high))
        weight_semantic, weight_detail, weight_current = self.mix_gate(s, d, c)
        branch_prior = torch.softmax(self.branch_logits, dim=0)
        out = self.fuse(
            torch.cat(
                [
                    (weight_detail * branch_prior[0]) * d,
                    (weight_current * branch_prior[1]) * c,
                    (weight_semantic * branch_prior[2]) * s,
                ],
                dim=1,
            )
        )
        return c + torch.sigmoid(self.residual_alpha) * out

# ----------------------------
# Innovation-2: FDSG
# ----------------------------
class FDSG(nn.Module):
    def __init__(self, c, level: int, r=8):
        super().__init__()
        self.level = level
        cr = max(16, c // r)

        self.low = nn.AvgPool2d(3, 1, 1)
        self.reduce = nn.Sequential(
            nn.Conv2d(c, cr, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(cr, c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        self.high_refine = nn.Sequential(
            nn.Conv2d(cr, cr, 3, 1, 1, groups=cr, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True),
        )
        self.low_refine = nn.Sequential(
            nn.Conv2d(cr, cr, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True),
        )

        self.content_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cr, max(8, cr // 4), 1, 1, 0),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, cr // 4), 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(cr, cr, 3, 1, 1, groups=cr, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True),
            nn.Conv2d(cr, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

        prior = {3: 0.70, 4: 0.55, 5: 0.35, 6: 0.30}.get(level, 0.5)
        self.register_buffer("prior", torch.tensor(prior).float().view(1, 1, 1, 1))
        self.gate_logits = nn.Parameter(
            torch.tensor([FDSG_GATE_CONTENT_WEIGHT, FDSG_GATE_SPATIAL_WEIGHT, FDSG_GATE_PRIOR_WEIGHT], dtype=torch.float)
        )
        self.prior_bias = nn.Parameter(torch.tensor(0.0))
        self.residual_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        xr = self.reduce(x)
        low_base = self.low(xr)
        low = self.low_refine(low_base)
        high = self.high_refine(xr - low_base)
        gc = self.content_gate(xr).to(dtype=x.dtype)
        gs = self.spatial_gate(torch.abs(high - low)).to(dtype=x.dtype)
        prior = self.prior.to(device=x.device, dtype=x.dtype)
        texture_score = torch.sigmoid(torch.mean(torch.abs(xr - low_base), dim=1, keepdim=True)).to(dtype=x.dtype)
        adaptive_prior = torch.clamp(prior + self.prior_bias.tanh() * (texture_score - 0.5), 0.0, 1.0)
        gate_w = torch.softmax(self.gate_logits, dim=0).to(device=x.device, dtype=x.dtype)
        g = torch.clamp(
            gate_w[0] * gc + gate_w[1] * gs + gate_w[2] * adaptive_prior,
            GATE_MIN_VALUE,
            GATE_MAX_VALUE,
        )
        out = g * high + (1.0 - g) * low
        return x + torch.sigmoid(self.residual_alpha) * self.expand(out)
# ----------------------------
# Innovation-3: DetectGLR
# ----------------------------
# Ultralytics 🚀 AGPL-3.0



class DetectGLR(Detect):
    """
    Detect + Gradient-aware Layer Reweight (GLR)
    """

    def __init__(self, nc=80, ch=()):
        if ch is None:
            ch = ()
        if not isinstance(ch, (list, tuple)):
            raise TypeError(f"Invalid ch type: {type(ch)}, ch={ch}")
        if len(ch) == 0:
            raise ValueError(
                "DetectGLR got empty ch. "
                "Please check yolo11.yaml head args and parse_model argument order."
            )

        super().__init__(nc=nc, ch=ch)

        self.momentum = 0.9
        self.eps = 1e-6
        self.warmup_iters = GLR_WARMUP_ITERATIONS
        self.reweight_power = GLR_REWEIGHT_POWER
        self.register_buffer("ema_cls", torch.ones(self.nl))
        self.register_buffer("ema_box", torch.ones(self.nl))
        self.register_buffer("iters", torch.zeros(1))

    @torch.no_grad()
    def update_glr(self, grad_cls: torch.Tensor, grad_box: torch.Tensor):
        if grad_cls.numel() != self.nl or grad_box.numel() != self.nl:
            return
        g_cls = torch.nan_to_num(grad_cls.detach().to(self.ema_cls.device), nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0)
        g_box = torch.nan_to_num(grad_box.detach().to(self.ema_box.device), nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0)
        mean_cls = g_cls.mean().clamp_min(NORMALIZATION_MIN_MEAN)
        mean_box = g_box.mean().clamp_min(NORMALIZATION_MIN_MEAN)
        g_cls = g_cls / mean_cls
        g_box = g_box / mean_box
        self.ema_cls.mul_(self.momentum).add_(g_cls * (1 - self.momentum))
        self.ema_box.mul_(self.momentum).add_(g_box * (1 - self.momentum))
        self.iters.add_(1)

    def glr_weights(self):
        ones = torch.ones_like(self.ema_cls)
        if self.training:
            ramp = torch.clamp((self.iters - self.warmup_iters) / max(float(self.warmup_iters), 1.0), 0.0, 1.0).to(
                self.ema_cls.device
            )
        else:
            ramp = self.ema_cls.new_tensor(1.0)
        ema_cls = torch.nan_to_num(self.ema_cls, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(self.eps)
        ema_box = torch.nan_to_num(self.ema_box, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(self.eps)
        inv_cls = ema_cls.pow(-self.reweight_power)
        inv_box = ema_box.pow(-self.reweight_power)
        a_cls = inv_cls / (inv_cls.sum() + self.eps) * self.nl
        a_box = inv_box / (inv_box.sum() + self.eps) * self.nl
        a_cls = a_cls.clamp(0.25, 4.0)
        a_box = a_box.clamp(0.25, 4.0)
        a_cls = a_cls / (a_cls.sum() + self.eps) * self.nl
        a_box = a_box / (a_box.sum() + self.eps) * self.nl
        return (1 - ramp) * ones + ramp * a_cls, (1 - ramp) * ones + ramp * a_box