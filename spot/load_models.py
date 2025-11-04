import torch
import timm
from . import nn
from . import patch_dropout
from torchvision.transforms import v2
from torch.nn import Identity


valid_models = [
    'spot_mae_b16',
    'spot_21k_b16',
    'spot_1k_b16',
    'vit_mae_b16',
    'vit_21k_b16',
    'vit_1k_b16'
]

def load_trained_model(
    model_name: str, path:str|None=None,
    sampler:str='grid', ksize:int=16, n_features:int=196,
    force_hub_reload:bool=False
):
    assert model_name in valid_models, f'Invalid model name: {model_name}. Valid models are: {valid_models}'
    spotpaths = dict(
        spot_mae_b16='checkpoints/spot_mae_b16.pth',
        spot_21k_b16='checkpoints/spot_21k_b16.pth',
        spot_1k_b16='checkpoints/spot_1k_b16.pth',
    )
    
    timm_names = dict(
        vit_21k_b16='vit_base_patch16_224.augreg2_in21k_ft_in1k',
        vit_1k_b16='vit_base_patch16_224.augreg_in1k',
    )
    use_timm = False
    use_mae = False
    if model_name not in spotpaths:
        if 'mae' not in model_name:
            use_timm = True
        else:
            use_mae = True

    if use_timm:
        model = timm.create_model(timm_names[model_name], pretrained=True)
        if n_features < 196:
            model.patch_drop = patch_dropout.PatchDropout(1-n_features/196)
        return model.eval()
    elif use_mae:
        model = torch.hub.load(
            'xxxx', # NB: Will be available in released repo
            'mae_vit_base_patch16_in1k',
            pretrained=True,
            global_pool=True,
            source='github',
            force_reload=force_hub_reload
        )
        if n_features < 196:
            model.patch_drop = patch_dropout.PatchDropout(1-n_features/196) # type: ignore
        return model.eval() # type: ignore
    
    kwargs = {'depth': 12, 'embed_dim': 768, 'heads': 12, 'dop_path': 0.0}
    kwargs['ksize'] = ksize
    kwargs['n_features'] = n_features
    kwargs['n_classes'] = 1000
    kwargs['sampler'] = sampler

    specifics = dict(
        spot_21k_b16=dict(
            qkv_bias=True,
            learnable_logprior=False,
            logprior=None,
            lnqk=False,
            pre_norm=False,
        ),
        spot_1k_b16=dict(
            qkv_bias=True,
            learnable_logprior=False,
            logprior=None,
            lnqk=False,
            pre_norm=False,
        ),
        spot_mae_b16=dict(
            qkv_bias=True,
            learnable_logprior=False,
            logprior=None,
            lnqk=False,
            pre_norm=False,
            global_pool=True,
        ),
    )[model_name]
    kwargs = {**kwargs, **specifics}
    model = nn.SPoTClassifier(**kwargs)
    
    path = path if path is not None else spotpaths[model_name]
    state_dict = torch.load(path, map_location='cpu', weights_only=False)
    
    if 'model' in state_dict:
        state_dict = state_dict['model']    
    
    if 'tokenizer.logprior' in state_dict:
        del state_dict['tokenizer.logprior']
    
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def get_validation_transform(model_name:str, imgsize:int = 224, crop_pct:float = 0.875):
    assert model_name in valid_models, f'Invalid model name: {model_name}. Valid models are: {valid_models}'
    preimgsize = int(round(imgsize / crop_pct))
    use_standard_transform = False
    if 'spot' in model_name:
        use_standard_transform = True
    elif 'mae' in model_name: 
        use_standard_transform = True

    if use_standard_transform:
        normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
    else:
        normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    return v2.Compose([
        v2.RandomResizedCrop((preimgsize, preimgsize), (1,1), interpolation=3),
        v2.CenterCrop((imgsize, imgsize)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        normalize
    ])

    
