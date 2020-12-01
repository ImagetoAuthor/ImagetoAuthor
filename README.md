1.主干网络：EfficientNet-B3。尝试过resnet18,50,densenet,resnest等作为主干，b3效果最好。最后一层添加maxpool,和avgpool进行拼接，增加特征表示。最后发现，针对此数据集，只用avgpool效果更佳。

2.loss:ce和focalloss，ce+labelsmooth效果最佳。

3.数据增强：mixup和随机擦除。

4.其他trick：tta和投票。

5.optimizer:Adam：3e-4, SGD：1e-5.



看图识别作者模型:

在main.py文件中，先用mixup进行训练，用Adam进行优化。




