# --coding:utf-8 --
"""
# Author     ：comi
# version    ：python 3.8
# Description：
"""

from configs import config
from utils.evaluateBase import evaluateBase
from utils.helper import getAllAttrs
from utils.exporter import Exporter

class evaluateLuna(evaluateBase):

    def __init__(self, model_lists, labels):
        super(evaluateLuna, self).__init__(model_lists)
        self.pth_path = config.pth_luna_path
        self.dataset = 'luna'
        if self.mode == '2d':
            self.seg_path = config.seg_path_luna_2d
        else:
            self.seg_path = config.seg_path_luna_3d
        self.exporter = Exporter(self.seg_path)
        self.run(labels)


if __name__ == '__main__':
    config.train = False

    loss_lists = ['dice', 'bce', 'focal']  #
    model2d = ['unet', 'unetpp', 'unet3p',
            #    'raunet', 'cpfnet',  'sgunet', 'bionet',
            #    'uctransnet', 'utnet', 'swinunet', 'unext'
               ]
    
    model3d = ['unet', 
            #    'resunet', 'vnet', 'ynet', 'unetpp', 'reconnet', 'transbts', 'wingsnet', 'unetr', 
               ]


    mode = config.mode
    if mode == '2d':
        evaluateLuna(model2d, None, ).to(config.device)  # 整体评估，读入全部数据
    else:
        evaluateLuna(model3d, None, ).to(config.device)  # 整体评估，读入全部数据

    # for labels in getAllAttrs(True).values():  # todo 分项评估
    #     evaluateLuna(model3d, labels).to(config.device)
