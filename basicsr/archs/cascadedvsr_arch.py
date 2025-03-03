import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import ResidualBlocksWithInputConv, PixelShufflePack
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.memory_util import *
from einops import rearrange
import numbers
from basicsr.archs.cbam import CBAM
from tqdm import tqdm

def read_momory(current_key, current_selection, memory_key, memory_shrinkage, memory_value):
        """
        current_key : b, c, h, w
        current_key : b, c, h, w
        memory_key: b, thw, c
        memory_shrinkage: b, thw, 1
        memory_value: b, c, thw
        """
        _, _, h, w = current_key.shape
        affinity = get_affinity(memory_key, memory_shrinkage, current_key, current_selection)
        memory = readout(affinity, memory_value)
        # import pdb
        # pdb.set_trace()
        memory = rearrange(memory, 'b c (h w) -> b c h w', h=h, w=w)

        return memory

class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))

        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g

class FeatureFusionBlock(nn.Module):
    def __init__(self, mid_channels): # 1024, 256, 512, 512
        super().__init__()

        # self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(3*mid_channels, mid_channels)
        self.attention = CBAM(mid_channels)
        self.block2 = GroupResBlock(mid_channels, mid_channels)

    def forward(self, x, g):
        """
        x: B * 1024 * 1/16H * 1/16W
        g: B * num_objects * 256 * 1/16H * 1/16W
        """
        # batch_size, t = g.shape[:2]

        g = torch.cat([x,g], dim=1) # B * num_objects * (1024+256) * 1/16H * 1/16W
        g = self.block1(g) # B * num_objects * 512 * 1/16H * 1/16W
        r = self.attention(g) # B x num_objects * 512 * 1/16H * 1/16W

        g = self.block2(g+r) # B * num_objects * 512 * 1/16H * 1/16W

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, mid_dim):
        super().__init__()
        self.hidden_dim = mid_dim

        self.transform = nn.Conv2d(2*mid_dim, mid_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):

        g = torch.cat([g, h], 1) # B * num_objects * (mid_dim+hidden_dim) * 1/16H * 1/16W
        values = self.transform(g) # B * num_objects * hidden_dim*3 * 1/16H * 1/16W
        forget_gate = torch.sigmoid(values[:,:self.hidden_dim]) # B * num_objects * hidden_dim * 1/16H * 1/16W
        update_gate = torch.sigmoid(values[:,self.hidden_dim:self.hidden_dim*2]) # B * num_objects * hidden_dim * 1/16H * 1/16W
        new_value = torch.tanh(values[:,self.hidden_dim*2:]) # B * num_objects * hidden_dim * 1/16H * 1/16W
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value # B * num_objects * hidden_dim * 1/16H * 1/16W

        return new_h

class FeatureFusion(nn.Module):
    def __init__(self, mid_channels):
        super().__init__()
        self.fuser = FeatureFusionBlock(mid_channels)
        self.hidden_update = HiddenUpdater(mid_channels)

    def forward(self, feat, hidden, memory):
        g = self.fuser(feat, torch.cat([memory, hidden], 1)) # b,c,h,w
        hidden = self.hidden_update(g, hidden)

        return g, hidden

@ARCH_REGISTRY.register()
class CasMEMVSR(nn.Module):
    """Enhanced BasicVSR network structure.
    Support either x4 upsampling or same size output.
    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        extract_blocks (int, optional): The number of residual blocks in feature
            extraction module. Default: 1.
        propagation_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 15.
        recons_blocks (int, optional): The number of residual blocks in reconstruction
            module. Default: 3.
        propagation_branches (list[str], optional): The names of the propagation branches.
            Default: ('backward_1', 'forward_1').
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 extract_blocks=3,
                 propagation_blocks=10,
                 recons_blocks=3,
                 GFN_blocks=2,
                 is_low_res_input=True,
                 propagation_branches=['forward_1'],
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, num_blocks=extract_blocks)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, num_blocks=extract_blocks))
            
        self.backbone = nn.ModuleDict()
        self.short_term = nn.ModuleDict()
        self.fusion = nn.ModuleDict()
        self.propagation_branches = propagation_branches
        for i, module in enumerate(self.propagation_branches):
            self.backbone[module] = ResidualBlocksWithInputConv(
                (3+i) * mid_channels, mid_channels, num_blocks=propagation_blocks)
            self.short_term[module] = CasBiGDFN(dim=mid_channels, n_levels=4, ffn_expansion_factor=2, bias=True, LayerNorm_type='withbias', num_blocks=GFN_blocks)
            self.fusion[module] = FeatureFusion(mid_channels)

        # reconstruction module
        self.reconstruction = ResidualBlocksWithInputConv(
            (len(self.propagation_branches)*2+1) * mid_channels, mid_channels, num_blocks=recons_blocks)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lqs):
        """Forward function for BasicVSR++.
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        feats = {}
        slid_for = {}

        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                _feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(_feat)
                torch.cuda.empty_cache()
        else:
            _feats = self.feat_extract(lqs.view(-1, c, h, w))
            _feats = _feats.view(n, t, -1, _feats.size(2), _feats.size(3))
            feats['spatial'] = [_feats[:, i, :, :, :] for i in range(0, t)]

        # feature propgation
        for module in self.propagation_branches:
            feats[module] = []
            slid_for[module] = []


            feats, slid_for = self.propagate(feats, slid_for, module)

            if self.cpu_cache:
                # del flows
                torch.cuda.empty_cache()
                
        out = self.reconstruct(lqs, feats, slid_for)

        return out


    def propagate(self, feats, slid_for, branch_name):
        """Propagate the latent features throughout the sequence.
        Args:
            feats (Dict[list[tensor]]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            branch_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        """

        n, _, h, w = feats['spatial'][0].size()
        t = len(feats['spatial'])

        fusion = []

        frame_idx = range(0, t) # 0,1,...,t-1
        if 'backward' in branch_name:
            frame_idx = frame_idx[::-1]
            # flow_idx = frame_idx

        feat_prop = feats['spatial'][0].new_zeros(n, self.mid_channels, h, w) # (n, 64, h, w)

        for i in range(t):
            feat_current = feats['spatial'][frame_idx[i]]
            if i == 0 :
                feat_past = feat_current
                feat_future = feats['spatial'][frame_idx[i+1]]
            elif i == t-1:
                feat_past = feats['spatial'][frame_idx[i-1]]
                feat_future = feat_current

            else:
                feat_past = feats['spatial'][frame_idx[i-1]]
                feat_future = feats['spatial'][frame_idx[i+1]]

            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()


            if i == 0:
                feat_l = [feat_current] +[
                    feats[k][frame_idx[i]] for k in feats if k not in ['spatial', branch_name]
                ] + [feat_prop] + [feat_future]

            else:
                feat_l = [feat_current] + [
                    feats[k][frame_idx[i]] for k in feats if k not in ['spatial', branch_name]
                ] + [feat_prop] + [fusion[-1]]


            if self.cpu_cache:
                feat_l = [f.cuda() for f in feat_l]

            feat_prop = feat_prop + self.backbone[branch_name](torch.cat(feat_l, dim=1))

            feat_for = self.short_term[branch_name](feat_past, feat_prop, feat_future)

            feat_fusion, feat_prop = self.fusion[branch_name](feat_for, feat_prop, feat_for)

            feats[branch_name].append(feat_prop)
            slid_for[branch_name].append(feat_for)
            fusion.append(feat_fusion)

            if self.cpu_cache:
                feats[branch_name][-1] = feats[branch_name][-1].cpu()
                torch.cuda.empty_cache() # save the last feature

        return feats, slid_for


    def reconstruct(self, lqs, feats, slid_for):
        """Compute the output image given the features.
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.
        """
        outputs = []
        t = len(feats['spatial'])

        for i in range(t):
            # feat_l = [feats[k].pop(0) for k in feats]
            feat_l = [feats[k].pop(0) for k in feats] + [slid_for[k].pop(0) for k in slid_for]
            hr = torch.cat(feat_l, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SFGDFN_block(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type='withbias') -> None:
        super().__init__()
        self.channel = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = GDFN(dim, ffn_expansion_factor, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

class CasBiGDFN(nn.Module):
    def __init__(self, dim, n_levels=4, ffn_expansion_factor=2, bias=True, LayerNorm_type='withbias', num_blocks=2) -> None:
        super().__init__()
        self.channel = dim

        self.past_attn = nn.Sequential(*[SFGDFN_block(dim, n_levels, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks)])

        self.hidden_attn = nn.Sequential(*[SFGDFN_block(dim, n_levels, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks)])
        self.future_attn = nn.Sequential(*[SFGDFN_block(dim, n_levels, ffn_expansion_factor, bias, LayerNorm_type) for _ in range(num_blocks)])


    def forward(self, feature_past, hidden, feature_future):

        feature_past = self.past_attn(feature_past)
        hidden = self.hidden_attn(hidden+feature_past)
        feature_future = self.future_attn(feature_future+hidden)
        feature_1 = hidden * torch.sigmoid(feature_past)
        feature_2 = hidden * torch.sigmoid(feature_future)
        feature = feature_1 + feature_2

        return feature


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, ActivationCountAnalysis
    
    model = CasMEMVSR(mid_channels=64, extract_blocks=3, propagation_blocks=5, recons_blocks=3, GFN_blocks=2, propagation_branches=['forward_1']).cuda()
    CUDA_VISIBLE_DEVICES = 1

    scale = 4
    h, w = 1280, 720 # 1920, 1080

    x = torch.randn(1, 5, 3, h // scale, w // scale).cuda()

    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)


    num_frame = 5
    clip = 1

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    runtime = 0

    dummy_input =  torch.randn((1, num_frame, 3, 1280 // scale, 720 // scale)).cuda()
    # warm_up
    model.eval()
    with torch.no_grad():
      for _ in tqdm(range(clip)):
          _ = model(dummy_input)

      for _ in tqdm(range(clip)):
          start.record()
          _ = model(dummy_input)
          end.record()
          torch.cuda.synchronize()
          runtime += start.elapsed_time(end)

      per_frame_time = runtime / (num_frame * clip)

      print(f'{num_frame * clip} Number Frames x{scale}SR Per Frame Time: {per_frame_time:.6f} ms')
      print(f' x{scale}SR FPS: {(1000 / per_frame_time):.6f} FPS')
"""

| module                                             | #parameters or shape   | #flops     | #activations   |
|:---------------------------------------------------|:-----------------------|:-----------|:---------------|
| model                                              | 2.255M                 | 0.94T      | 3.277G         |
|  feat_extract.main                                 |  0.223M                |  64.199G   |  0.129G        |
|   feat_extract.main.0                              |   1.792K               |   0.498G   |   18.432M      |
|    feat_extract.main.0.weight                      |    (64, 3, 3, 3)       |            |                |
|    feat_extract.main.0.bias                        |    (64,)               |            |                |
|   feat_extract.main.2                              |   0.222M               |   63.701G  |   0.111G       |
|    feat_extract.main.2.0                           |    73.856K             |    21.234G |    36.864M     |
|    feat_extract.main.2.1                           |    73.856K             |    21.234G |    36.864M     |
|    feat_extract.main.2.2                           |    73.856K             |    21.234G |    36.864M     |
|  backbone.forward_1.main                           |  0.48M                 |  0.138T    |  0.203G        |
|   backbone.forward_1.main.0                        |   0.111M               |   31.85G   |   18.432M      |
|    backbone.forward_1.main.0.weight                |    (64, 192, 3, 3)     |            |                |
|    backbone.forward_1.main.0.bias                  |    (64,)               |            |                |
|   backbone.forward_1.main.2                        |   0.369M               |   0.106T   |   0.184G       |
|    backbone.forward_1.main.2.0                     |    73.856K             |    21.234G |    36.864M     |
|    backbone.forward_1.main.2.1                     |    73.856K             |    21.234G |    36.864M     |
|    backbone.forward_1.main.2.2                     |    73.856K             |    21.234G |    36.864M     |
|    backbone.forward_1.main.2.3                     |    73.856K             |    21.234G |    36.864M     |
|    backbone.forward_1.main.2.4                     |    73.856K             |    21.234G |    36.864M     |
|  short_term.forward_1                              |  0.331M                |  92.897G   |  1.991G        |
|   short_term.forward_1.past_attn                   |   0.11M                |   30.966G  |   0.664G       |
|    short_term.forward_1.past_attn.0                |    55.168K             |    15.483G |    0.332G      |
|    short_term.forward_1.past_attn.1                |    55.168K             |    15.483G |    0.332G      |
|   short_term.forward_1.hidden_attn                 |   0.11M                |   30.966G  |   0.664G       |
|    short_term.forward_1.hidden_attn.0              |    55.168K             |    15.483G |    0.332G      |
|    short_term.forward_1.hidden_attn.1              |    55.168K             |    15.483G |    0.332G      |
|   short_term.forward_1.future_attn                 |   0.11M                |   30.966G  |   0.664G       |
|    short_term.forward_1.future_attn.0              |    55.168K             |    15.483G |    0.332G      |
|    short_term.forward_1.future_attn.1              |    55.168K             |    15.483G |    0.332G      |
|  fusion.forward_1                                  |  0.554M                |  0.159T    |  0.148G        |
|   fusion.forward_1.fuser                           |   0.333M               |   95.58G   |   92.449M      |
|    fusion.forward_1.fuser.block1                   |    0.258M              |    74.318G |    55.296M     |
|    fusion.forward_1.fuser.attention                |    0.679K              |    28.229M |    0.289M      |
|    fusion.forward_1.fuser.block2                   |    73.856K             |    21.234G |    36.864M     |
|   fusion.forward_1.hidden_update.transform         |   0.221M               |   63.701G  |   55.296M      |
|    fusion.forward_1.hidden_update.transform.weight |    (192, 128, 3, 3)    |            |                |
|    fusion.forward_1.hidden_update.transform.bias   |    (192,)              |            |                |
|  reconstruction.main                               |  0.332M                |  95.551G   |  0.129G        |
|   reconstruction.main.0                            |   0.111M               |   31.85G   |   18.432M      |
|    reconstruction.main.0.weight                    |    (64, 192, 3, 3)     |            |                |
|    reconstruction.main.0.bias                      |    (64,)               |            |                |
|   reconstruction.main.2                            |   0.222M               |   63.701G  |   0.111G       |
|    reconstruction.main.2.0                         |    73.856K             |    21.234G |    36.864M     |
|    reconstruction.main.2.1                         |    73.856K             |    21.234G |    36.864M     |
|    reconstruction.main.2.2                         |    73.856K             |    21.234G |    36.864M     |
|  upsample1.upsample_conv                           |  0.148M                |  42.467G   |  73.728M       |
|   upsample1.upsample_conv.weight                   |   (256, 64, 3, 3)      |            |                |
|   upsample1.upsample_conv.bias                     |   (256,)               |            |                |
|  upsample2.upsample_conv                           |  0.148M                |  0.17T     |  0.295G        |
|   upsample2.upsample_conv.weight                   |   (256, 64, 3, 3)      |            |                |
|   upsample2.upsample_conv.bias                     |   (256,)               |            |                |
|  conv_hr                                           |  36.928K               |  0.17T     |  0.295G        |
|   conv_hr.weight                                   |   (64, 64, 3, 3)       |            |                |
|   conv_hr.bias                                     |   (64,)                |            |                |
|  conv_last                                         |  1.731K                |  7.963G    |  13.824M       |
|   conv_last.weight                                 |   (3, 64, 3, 3)        |            |                |
|   conv_last.bias                                   |   (3,)                 |            |                |
|  img_upsample                                      |                        |  55.296M   |  0             |
torch.Size([1, 5, 3, 1280, 720])


"""