1.主干网络：EfficientNet-B3。尝试过resnet18,50,densenet,resnest等作为主干，b3效果最好。最后一层添加maxpool,和avgpool进行拼接，增加特征表示。最后发现，针对此数据集，只用avgpool效果更佳。

2.loss:ce和focalloss，ce+labelsmooth效果最佳。

3.数据增强：mixup和随机擦除。

4.其他trick：tta和投票。

5.optimizer:Adam：3e-4, SGD：1e-5.



运行:

1.修改data路径，在main.py文件中，先用mixup进行训练，用Adam进行优化。

2.修改路径运行test.py，对mixup结果进行测试。

3.加载mixup 训练后的pth，用SGD使用原数据再训练几轮，这样能进一步提高。

4.修改csv文件路径，运行ART_Vote.py将结果最好的5组进行投票，得到最后的成绩。



