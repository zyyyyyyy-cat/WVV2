import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.vit_pytorch import vit_base_in, vit_ics_lup
from .visa_adapt import ETGM, DLFM, make_retrieval_feature

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.adapt_mode = cfg.MODEL.VISA_ADAPT.MODE
        if self.adapt_mode not in ('off', 'token_only', 'full'):
            raise ValueError('VISA_ADAPT.MODE must be off, token_only, or full')
        if self.adapt_mode == 'full' and pretrain_choice != 'none':
            raise ValueError('WV V2 full mode requires original UAD Epoch 120 initialization')
        self.use_specific = self.adapt_mode in ('token_only', 'full')
        if self.use_specific and list(cfg.DATASETS.MODALITIES) != ['RGB', 'IR', 'Thermal']:
            raise ValueError('adapted WHU training expects DATASETS.MODALITIES = [RGB, IR, Thermal]')
        self.fusion_beta = cfg.MODEL.VISA_ADAPT.FUSION_BETA

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base=factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        specific_tokens=self.use_specific)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained model......from {}'.format(model_path))
        if self.adapt_mode == 'full':
            for parameter in self.base.fc.parameters():
                parameter.requires_grad_(False)

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if self.adapt_mode == 'full':
            adapt = cfg.MODEL.VISA_ADAPT
            self.etgm = ETGM(self.in_planes, adapt.EXPERTS, adapt.TOP_K,
                             adapt.PROMPT_LEN, adapt.HEADS)
            self.dlfm = DLFM(self.in_planes, adapt.HEADS, adapt.GRAPH_K)
            self.local_classifier = nn.Parameter(torch.empty(self.num_classes, self.in_planes))
            nn.init.normal_(self.local_classifier, std=.01)
        
    def forward(self, x=None, label=None, camids=None, mode=0, return_components=False):
        if self.use_specific:
            if mode == 0:
                if len(x) != 3:
                    raise ValueError('adapted training requires RGB, NIR, TIR in this order')
                modality_ids = torch.cat([torch.full((part.shape[0],), i, dtype=torch.long,
                                                     device=part.device) for i, part in enumerate(x)])
                images = torch.cat(list(x), dim=0)
                cameras = torch.cat(list(camids), dim=0) if camids is not None else None
            else:
                images = x
                # Dataset IDs are 1=RGB, 2=NIR, 3=TIR; mode is an evaluation group.
                if label is None:
                    raise ValueError('evaluation requires dataset modality ids via label')
                modality_ids = torch.as_tensor(label, device=x.device, dtype=torch.long) - 1
                if modality_ids.shape != (x.shape[0],):
                    raise ValueError('one modality id is required for each evaluation image')
                cameras = camids
            # WHU's fourth metadata field (view in the original backbone) is
            # the zero-based modality index when SIE_VIEW is enabled.
            view_ids = modality_ids if self.base.view_num > 0 else None
            shared, specific, patch11 = self.base(images, cam_label=cameras,
                                                       modal_label=modality_ids, view_label=view_ids)
            if mode == 0:
                neck_feature = self.bottleneck(shared)
                score = self.classifier(neck_feature)
                if self.adapt_mode == 'token_only':
                    return score, shared, neck_feature
                query_shared, query_specific, balance = self.etgm(shared, specific, modality_ids)
                local = self.dlfm(shared, specific, patch11, query_shared, query_specific)
                with torch.autocast(device_type=local.device.type, enabled=False):
                    local_score = 16.0 * F.linear(F.normalize(local.float(), dim=1),
                                                   F.normalize(self.local_classifier.float(), dim=1))
                return score, shared, specific, local_score, local, balance
            if self.adapt_mode == 'token_only':
                return shared
            query_shared, query_specific, _ = self.etgm(shared, specific, modality_ids)
            local = self.dlfm(shared, specific, patch11, query_shared, query_specific)
            retrieval = make_retrieval_feature(shared, local, self.fusion_beta)
            if return_components:
                return {'shared': torch.nn.functional.normalize(shared.float(), dim=1),
                        'local': torch.nn.functional.normalize(local.float(), dim=1),
                        'fusion': retrieval}
            return retrieval
        if mode==0:
            x = torch.cat(list(x), dim=0)
            global_feat = self.base(x)
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return cls_score, global_feat, feat
        else:
            global_feat = self.base(x)
            feat = self.bottleneck(global_feat)
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu', weights_only=False)
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if self.use_specific:
            self.load_state_dict({(key[7:] if key.startswith('module.') else key): value
                                  for key, value in param_dict.items()}, strict=True)
            print('Loaded complete adapted model from {}'.format(trained_path))
            return
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_uad_checkpoint(self, trained_path):
        """Restore every UAD identity parameter; only V2 additions may be absent."""
        checkpoint = torch.load(trained_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        source = {(key[7:] if key.startswith('module.') else key): value
                  for key, value in state.items()}
        target = self.state_dict()
        old_keys = {key for key in target if key.startswith(('base.', 'bottleneck.', 'classifier.'))
                    and not key.startswith(('base.specific_tokens.', 'base.specific_pos_embed'))}
        missing = sorted(old_keys - source.keys())
        mismatched = sorted(key for key in old_keys & source.keys()
                            if source[key].shape != target[key].shape)
        if missing or mismatched:
            raise RuntimeError(f'UAD checkpoint mismatch: missing={missing}, shapes={mismatched}')
        unexpected = sorted(key for key in source if key not in target)
        if unexpected:
            raise RuntimeError(f'Unexpected UAD checkpoint keys: {unexpected}')
        self.load_state_dict({key: source[key] for key in old_keys}, strict=False)
        print(f'Loaded {len(old_keys)} complete UAD tensors from {trained_path}')
        return {'loaded': sorted(old_keys), 'new': sorted(set(target) - old_keys)}

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_in': vit_base_in,
    'vit_ics_lup': vit_ics_lup,
}

def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    return model
