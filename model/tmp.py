import torch


def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    softmax_version=False,
    nrows = -1,
    delta_softplus = True,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    xs = CrossScan.apply(x)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
        
    # to enable fvcore.nn.jit_analysis: inputs[i].debugName
    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, H, W)
    
    y = CrossMerge.apply(ys)

    if softmax_version:
        y = y.softmax(y, dim=-1).to(x.dtype)
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y).to(x.dtype)
    
    return y

model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24, 
        channels=3, 
        num_classes=1,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type='mean',
        if_abs_pos_embed=True, 
        if_rope=True,
        if_rope_residual=True, 
        bimamba_type="v2",
        if_cls_token=True, 
        **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block



















# MODEL:
#   TYPE: vssm
#   NAME: vssm_tiny
#   DROP_PATH_RATE: 0.2
#   VSSM:
#     EMBED_DIM: 96
#     DEPTHS: [ 2, 2, 9, 2 ]
#     D_STATE: 16
#     DT_RANK: "auto"
#     SSM_RATIO: 2.0
#     MLP_RATIO: 0.0
#     DOWNSAMPLE: "v1"
def build_vssm_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=224, 
            in_chans=3, 
            num_classes=1, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            # ===================
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0,
            attn_drop_rate=0.0,
            shared_ssm=False,
            softmax_version=False,
            # ===================
            drop_rate=0.0,
            drop_path_rate=0.2,
            mlp_ratio=0.0,
            patch_norm=True,
            # norm_layer=nn.LayerNorm,
            downsample_version="v1",
            use_checkpoint=False,
        )
        return model

    return None
class BAAMamba(nn.Module):
    def __init__(
            self, 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            # =========================
            d_state=16, 
            dt_rank="auto", 
            ssm_ratio=2.0, 
            attn_drop_rate=0., 
            shared_ssm=False,
            softmax_version=False,
            # =========================
            drop_rate=0., 
            drop_path_rate=0.1, 
            mlp_ratio=4.0,
            patch_norm=True, 
            norm_layer=nn.LayerNorm,
            downsample_version: str = "v2",
            use_checkpoint=False,  
            **kwargs,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, self.embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(self.embed_dim) if patch_norm else nn.Identity()), 
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            if downsample_version == "v2":
                downsample = self._make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()
            else:
                downsample = PatchMerging2D(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                attn_drop_rate=attn_drop_rate,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        # ===========================
        d_state=16,
        dt_rank="auto",
        ssm_ratio=2.0,
        attn_drop_rate=0.0, 
        shared_ssm=False,
        softmax_version=False,
        # ===========================
        mlp_ratio=4.0,
        drop_rate=0.0,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
                **kwargs,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        print(f'after patch_embed: {x.shape}')
        import pdb; pdb.set_trace()
        x = CrossScan.apply(x)
        # import pdb; pdb.set_trace()
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x
#TODO
    # def flops(self, shape=(3, 224, 224)):
    #     # shape = self.__input_shape__[1:]
    #     supported_ops={
    #         "aten::silu": None, # as relu is in _IGNORED_OPS
    #         "aten::neg": None, # as relu is in _IGNORED_OPS
    #         "aten::exp": None, # as relu is in _IGNORED_OPS
    #         "aten::flip": None, # as permute is in _IGNORED_OPS
    #         "prim::PythonOp.CrossScan": None,
    #         "prim::PythonOp.CrossMerge": None,
    #         "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
    #         "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
    #     }

    #     model = copy.deepcopy(self)
    #     model.cuda().eval()

    #     input = torch.randn((1, *shape), device=next(model.parameters()).device)
    #     params = parameter_count(model)[""]
    #     Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

    #     del model, input
    #     return sum(Gflops.values()) * 1e9
    #     return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    
@register_model
def BAA_model_224(pretrained=False, **kwargs):
    model = BAAMamba(
        patch_size=224, 
            in_chans=3, 
            num_classes=1, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            # ===================
            d_state=16,
            dt_rank="auto",
            ssm_ratio=2.0,
            attn_drop_rate=0.0,
            shared_ssm=False,
            softmax_version=False,
            # ===================
            drop_rate=0.0,
            drop_path_rate=0.2,
            mlp_ratio=0.0,
            patch_norm=True,
            # norm_layer=nn.LayerNorm,
            downsample_version="v1",
            use_checkpoint=pretrained,
    )
    
    return model