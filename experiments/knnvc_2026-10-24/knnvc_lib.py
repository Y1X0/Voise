import os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import weight_norm
SR, HOP, LAYER, K = 16000, 320, 6, 4
W = os.environ.get("KNNVC_W", os.path.join(os.environ.get("VOISE_DL", "dl"), "knnvc_weights"))

# ---------------------------------------------------------------- WavLM-Large (HF module, fairseq weights)
def load_wavlm():
    from transformers import WavLMConfig, WavLMModel
    ck = torch.load(os.path.join(W, "WavLM-Large.pt"), map_location="cpu", weights_only=True)
    c = ck["cfg"]
    assert c["layer_norm_first"] and c["extractor_mode"] == "layer_norm" and c["normalize"]
    cfg = WavLMConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
                      feat_extract_norm="layer", do_stable_layer_norm=True, conv_bias=False, num_buckets=320,
                      max_bucket_distance=800, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
                      hidden_act="gelu", feat_proj_dropout=0.0, hidden_dropout=0.0, attention_dropout=0.0,
                      activation_dropout=0.0, layerdrop=0.0, apply_spec_augment=False)
    m = WavLMModel(cfg).eval()
    sd = {}
    for k, v in ck["model"].items():
        n = k
        for a, b in (("mask_emb", "masked_spec_embed"), (".0.weight", ".conv.weight"), (".2.1.", ".layer_norm."),
                     ("post_extract_proj", "feature_projection.projection"),
                     ("encoder.pos_conv.0.weight_g", "encoder.pos_conv_embed.conv.parametrizations.weight.original0"),
                     ("encoder.pos_conv.0.weight_v", "encoder.pos_conv_embed.conv.parametrizations.weight.original1"),
                     ("encoder.pos_conv.0.bias", "encoder.pos_conv_embed.conv.bias"),
                     (".self_attn.relative_attention_bias.", ".attention.rel_attn_embed."),
                     (".self_attn.grep_linear.", ".attention.gru_rel_pos_linear."),
                     (".self_attn.grep_a", ".attention.gru_rel_pos_const"),
                     (".self_attn_layer_norm.", ".layer_norm."), (".self_attn.", ".attention."),
                     (".fc1.", ".feed_forward.intermediate_dense."), (".fc2.", ".feed_forward.output_dense.")):
            if a in n and (a != ".0.weight" or n.startswith("feature_extractor")):
                n = n.replace(a, b)
        if k in ("layer_norm.weight", "layer_norm.bias"):
            n = "feature_projection." + k
        sd[n] = v
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return m


@torch.no_grad()
def wavlm_feats(m, wav, normalize=False, layer=LAYER):
    x = torch.as_tensor(np.asarray(wav, np.float32))[None]
    if normalize:
        x = F.layer_norm(x, x.shape)
    hs = m(x, output_hidden_states=True).hidden_states
    return hs[layer][0]                                          # output of transformer layer 6, [T, 1024]


# ---------------------------------------------------------------- prematched HiFi-GAN (knn-vc config v1)
def gp(k, d=1): return (k * d - d) // 2


class RB(nn.Module):
    def __init__(s, ch, k, ds):
        super().__init__()
        s.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=gp(k, d))) for d in ds])
        s.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, 1, padding=gp(k))) for _ in ds])

    def forward(s, x):
        for c1, c2 in zip(s.convs1, s.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
        return x


class Gen(nn.Module):
    def __init__(s):
        super().__init__()
        rates, ks, ch = (10, 8, 2, 2), (20, 16, 4, 4), 512
        s.lin_pre = nn.Linear(1024, 512)
        s.conv_pre = weight_norm(nn.Conv1d(512, ch, 7, 1, padding=3))
        s.ups = nn.ModuleList([weight_norm(nn.ConvTranspose1d(ch // 2 ** i, ch // 2 ** (i + 1), k, u, padding=(k - u) // 2))
                               for i, (u, k) in enumerate(zip(rates, ks))])
        s.resblocks = nn.ModuleList([RB(ch // 2 ** (i + 1), k, (1, 3, 5)) for i in range(4) for k in (3, 7, 11)])
        s.conv_post = weight_norm(nn.Conv1d(ch // 16, 1, 7, 1, padding=3))

    def forward(s, f):                                           # f [B, T, 1024]
        x = s.conv_pre(s.lin_pre(f).transpose(1, 2))
        for i, up in enumerate(s.ups):
            x = up(F.leaky_relu(x, 0.1))
            x = sum(s.resblocks[i * 3 + j](x) for j in range(3)) / 3
        return torch.tanh(s.conv_post(F.leaky_relu(x))).squeeze(1)


def load_gen():
    g = Gen().eval()
    sd = torch.load(os.path.join(W, "prematch_g_02500000.pt"), map_location="cpu", weights_only=True)["generator"]
    g.load_state_dict(sd, strict=True)
    return g


def knn(query, matching, k=K):
    q, mset = F.normalize(query, dim=-1), F.normalize(matching, dim=-1)
    idx = (q @ mset.T).topk(k, dim=-1).indices                   # cosine similarity
    return matching[idx].mean(1)


