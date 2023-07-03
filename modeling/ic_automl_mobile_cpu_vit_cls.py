# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from byted_nnflow.ic_automl_model.classification.utils.mobile_cpu_vit_cls_model_cfg import mobile_cpu_vit_cls_model_cfg
from byted_nnflow.ic_automl_model.classification.utils.mobile_op_pool import NNFlow_Mobile_OP_Pool
from byted_nnflow.ic_automl_model.classification.utils.mobile_vit_op_pool import MobileTrans_Pool
from byted_nnflow.ic_automl_model.classification.utils.model_utils import (Activation, ConvBNReLU, GlobalPow2Pool,
                                                                           Identity, load_checkpoint)
from byted_nnflow.search.torch_frame import NAS, MutableBlock


def mobile_cpu_vit_cls_arch_cfg(flops=100.0,
                                architecture=None,
                                model_size=None,
                                constrain1=False,
                                constrain2=False,
                                use_se=False,
                                light_se=False,
                                ir_block_shortcut=True,
                                op_type='NNFlowBottle_v1',
                                channel_div=8,
                                se_div=8,
                                sr_ratios=[1, 1, 1, 1, 1, 1, 1],
                                model_id='62b99b62'):
    if flops == 20.4:
        model_size = 2.3302
        architecture = [1, 6, 0, 0, 2, 3, 0, 0,
                        4, 6, 6, 6, 6,
                        8, 8, 4, 4, 4]
        op_type = ['NNFlowBottle_v1', 'NNFlowBottle_v1', 'NNFlowBottle_v1',
                   'NNFlowBottle_v1', 'MobileTrans_Pool_0_01']
        constrain1 = True
        model_id = '{{model_id}}'

    a_cfg = dict(model_size=model_size,
                 architecture=architecture,
                 constrain1=constrain1,
                 constrain2=constrain2,
                 use_se=use_se,
                 light_se=light_se,
                 ir_block_shortcut=ir_block_shortcut,
                 op_type=op_type,
                 channel_div=channel_div,
                 se_div=se_div,
                 sr_ratios=sr_ratios,
                 model_id=model_id)

    return a_cfg


class IC_AutoML_Mobile_CPU_ViT(nn.Module):

    def __init__(
            self,
            flops=100.0,
            user_name=None,
            image_channels=3,
            n_class=1000,
            dropout_rate=0.2,
            **kwargs):

        super(IC_AutoML_Mobile_CPU_ViT, self).__init__()
        self.version = 'v0.0.3.220526'
        self.flops = flops

        a_cfg = mobile_cpu_vit_cls_arch_cfg(self.flops)
        self.architecture = a_cfg.get('architecture')
        self.constrain1 = a_cfg.get('constrain1')
        self.constrain2 = a_cfg.get('constrain2')
        self.light_se = a_cfg.get('light_se')
        self.ir_block_shortcut = a_cfg.get('ir_block_shortcut')
        self.channel_div = a_cfg.get('channel_div')
        self.se_div = a_cfg.get('se_div')
        self.use_se = a_cfg.get('use_se')
        self.op_type = a_cfg.get('op_type')
        self.model_size = a_cfg.get('model_size')
        self.sr_ratios = a_cfg.get('sr_ratios')
        self.model_id = a_cfg.get('model_id')

        m_cfg = mobile_cpu_vit_cls_model_cfg(model_size=self.model_size, light_se=self.light_se)
        self.stride = m_cfg.get('stride')
        self.stage_repeats = m_cfg.get('stage_repeats')
        self.stage_out_channels = m_cfg.get('stage_out_channels')
        self.last_conv_out_channel = m_cfg.get('last_conv_out_channel')
        self.input_channel = m_cfg.get('input_channel')
        self.feature_out_index = m_cfg.get('feature_out_index')
        self.feature_out_channels = m_cfg.get('feature_out_channels')
        self.model_info = m_cfg.get('model_info')

        self.dropout_rate = dropout_rate
        self.act_func_first = Activation('hard_swish') if self.use_se else Activation('relu')
        self.use_avgpool_shortcut = False

        self.first_conv = ConvBNReLU(image_channels, self.input_channel, kernel_size=3, stride=2,
                                     act_func=self.act_func_first)
        self.out_stride = 2

        if self.model_size in [0.091, 0.092, 0.093, 0.094, 0.095, 0.05]:
            self.first_conv = ConvBNReLU(3, self.input_channel, stride=4, kernel_size=4,
                                         act_func=self.act_func_first)
            self.out_stride = 4

        block_id = 0
        features = []

        for stage_id in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[stage_id]
            output_channel = self.stage_out_channels[stage_id]

            if isinstance(self.op_type, list):
                op_type = self.op_type[stage_id]
            elif isinstance(self.op_type, str):
                op_type = self.op_type
            else:
                raise NotImplementedError

            for i in range(numrepeat):
                if self.stride[stage_id] == 2 and i == 0:
                    stride = 2
                    self.out_stride = self.out_stride * stride
                else:
                    stride = 1
                ######################################################
                if self.use_se:
                    act_func = Activation('hard_swish') if self.out_stride >= 16 else Activation('relu')
                    if self.light_se:
                        if self.flops > 100.0 and self.out_stride in [8, 16, 32] and \
                                (stage_id == len(self.stage_repeats) - 1 or self.stride[stage_id + 1] == 2):
                            use_se = True
                        elif self.flops < 100.0 and self.out_stride in [4, 16, 32] and \
                                (stage_id == len(self.stage_repeats) - 1 or self.stride[stage_id + 1] == 2):
                            use_se = True
                        else:
                            use_se = False
                    else:
                        if self.flops > 100.0 and self.out_stride in [8, 16, 32] and \
                                (stage_id == len(self.stage_repeats) - 1 or self.stride[stage_id + 1] == 2):
                            use_se = True
                        elif self.flops < 100.0 and self.out_stride in [4, 16, 32]:
                            use_se = True
                        else:
                            use_se = False
                else:
                    use_se = False
                    act_func = Activation('relu')
                ######################################################
                block_id += 1
                if 'Trans' in op_type:
                    image_height = 224
                    image_width = 224
                    height = image_height // self.out_stride
                    width = image_width // self.out_stride
                    op_pool = MobileTrans_Pool(input_height=height,
                                               input_width=width,
                                               op_type=op_type,
                                               input_channel=self.input_channel if i == 0 else output_channel,
                                               output_channel=output_channel,
                                               use_downsampling=True if stride == 2 else False,
                                               stride=stride,
                                               use_layer_norm=False,
                                               sr_ratio=self.sr_ratios[stage_id],
                                               **kwargs)
                else:
                    op_pool = NNFlow_Mobile_OP_Pool(self.input_channel,
                                                    output_channel,
                                                    stride=stride,
                                                    op_type=op_type,
                                                    constrain1=self.constrain1,
                                                    constrain2=self.constrain2,
                                                    act_func=act_func,
                                                    use_se=use_se,
                                                    se_div=self.se_div,
                                                    channel_div=self.channel_div,
                                                    ir_block_shortcut=self.ir_block_shortcut,
                                                    use_avgpool_shortcut=self.use_avgpool_shortcut)

                if len(op_pool) == 0:
                    raise NotImplementedError
                features.append(MutableBlock(op_pool))
                self.input_channel = output_channel

        self.features = nn.Sequential(*features)

        if self.model_size in [0.0911, 0.095, 0.093, 2.511, 2.512]:
            self.final_expand_layer = Identity()
        else:
            self.final_expand_layer = ConvBNReLU(self.stage_out_channels[-2], self.stage_out_channels[-1],
                                                 kernel_size=1, act_func=Activation('relu'))

        if self.use_se:
            self.avgpool = GlobalPow2Pool()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_mix_layer = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], self.last_conv_out_channel, bias=True),
            nn.BatchNorm1d(self.last_conv_out_channel),
            Activation(act_func='relu'),
        )
        if self.model_size in [0.32, 0.33, 0.092]:
            self.feature_mix_layer = Identity()

        # print("dropout_rate:", dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output = nn.Linear(self.last_conv_out_channel, n_class, bias=True)

        self = NAS.nnflow_nas_model_build(self,
                                          op_ids=self.architecture,
                                          flops=self.flops,
                                          user_name=user_name)
        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for name, module in self.named_modules():
            if hasattr(module, "is_bn_merged") and module.is_bn_merged is False:
                module.merge_bn()

    def forward_features(self, x):
        x = self.first_conv(x)
        out = list()
        if 0 in self.feature_out_index:
            out.append(x)
        for id, layer in enumerate(self.features):
            x = layer(x)
            if id + 1 in self.feature_out_index:
                out.append(x.contiguous())
        return out

    def forward(self, x):
        x = self.first_conv(x)
        # out = list()
        # if 0 in self.feature_out_index:
        #     out.append(x)
        for id, layer in enumerate(self.features):
            x = layer(x)
            # if id + 1 in self.feature_out_index:
            #     out.append(x.contiguous())
        x = self.final_expand_layer(x)
        # x = self.avgpool(x)
        # x = x.view(x.data.size(0), -1)
        # x = self.feature_mix_layer(x)
        # x = self.dropout(x)
        # x = self.output(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def get_IC_AutoML_Mobile_CPU_ViT(n_class=1000,
                                 image_channels=3,
                                 dropout_rate=0.2,
                                 flops=100.0,
                                 user_name=None,
                                 resume=None,
                                 **kwargs):
    net = IC_AutoML_Mobile_CPU_ViT(flops=flops,
                                   user_name=user_name,
                                   image_channels=image_channels,
                                   n_class=n_class,
                                   dropout_rate=dropout_rate,
                                   **kwargs)

    if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu')
        load_checkpoint(net, checkpoint, strict=False)
        print('load model success')

    return net


def main():
    flops = 20.4
    user_name = "licongcong.lufficc@bytedance.com"  # modify user_name to your email address
    resume = 'ic_automl_mobile_cpu_vit_cls_20.4M.pth.tar'  # None #

    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    net = get_IC_AutoML_Mobile_CPU_ViT(flops=flops,
                                       user_name=user_name,
                                       resume=resume,
                                       ff_dropout=0.0,
                                       attention_dropout=0.0,
                                       path_dropout=0.0,
                                       ).cuda()
    inp = torch.randn(1, 3, 224, 224).cuda()
    net(inp)


if __name__ == "__main__":
    main()
