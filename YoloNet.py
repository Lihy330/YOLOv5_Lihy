from Backbone import CSPDarkNet, Conv, CSPNet
import torch
import torch.nn as nn


class YOLO(nn.Module):
    def __init__(self, base_depth, base_channels, num_classes, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YOLO, self).__init__()
        self.backbone = CSPDarkNet(base_depth, base_channels)
        # 上采样2倍
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')
        self.feat3_conv_before_upSample = Conv(base_channels * 16, base_channels * 8, kernel_size=1, stride=1, padding=0)
        self.feat2_conv_before_upSample = Conv(base_channels * 8, base_channels * 4, kernel_size=1, stride=1, padding=0)
        self.feat3_upSample_feat2 = CSPNet(base_channels * 16, base_channels * 8, base_depth)
        self.feat2_upSample_feat1 = CSPNet(base_channels * 8, base_channels * 4, base_depth)
        # 下采样直接通过步长为2的卷积层实现
        self.feat1_downSample = Conv(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.feat1_downSample_feat2 = CSPNet(base_channels * 8, base_channels * 8, base_depth)
        self.feat2_downSample = Conv(base_channels * 8, base_channels * 8, kernel_size=3, stride=2, padding=1)
        self.feat2_downSample_feat3 = CSPNet(base_channels * 16, base_channels * 16, base_depth)
        # yolo_head
        self.head1 = nn.Conv2d(base_channels * 4, (num_classes + 5) * len(anchors_mask[0]), kernel_size=1, stride=1, padding=0)
        self.head2 = nn.Conv2d(base_channels * 8, (num_classes + 5) * len(anchors_mask[1]), kernel_size=1, stride=1, padding=0)
        self.head3 = nn.Conv2d(base_channels * 16, (num_classes + 5) * len(anchors_mask[2]), kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # 三个特征层shape分别为
        # [bs, 256, 80, 80]
        # [bs, 512, 40, 40]
        # [bs, 1024, 20, 20]
        feat1, feat2, feat3 = self.backbone(x)
        # [bs, 1024, 20, 20] => [bs, 512, 20, 20]
        out3 = self.feat3_conv_before_upSample(feat3)
        # [bs, 512, 20, 20] => [bs, 512, 40, 40]
        out3_upSample = self.upSample(out3)
        out2 = self.feat2_conv_before_upSample(self.feat3_upSample_feat2(torch.concat([feat2, out3_upSample], dim=1)))
        out2_upSample = self.upSample(out2)
        out1 = self.feat2_upSample_feat1(torch.concat([feat1, out2_upSample], dim=1))
        out1_downSample = self.feat1_downSample(out1)
        out2 = self.feat1_downSample_feat2(torch.concat([out2, out1_downSample], dim=1))
        out2_downSample = self.feat2_downSample(out2)
        out3 = self.feat2_downSample_feat3(torch.concat([out3, out2_downSample], dim=1))
        out1 = self.head1(out1)
        out2 = self.head2(out2)
        out3 = self.head3(out3)

        return out1, out2, out3


if __name__ == "__main__":
    img = torch.randn(1, 3, 640, 640)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img = img.to(device)
    num_classes = 20
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model = YOLO(3, 64, num_classes, anchors_mask)
    model = model.to(device)
    output = model(img)
    print(output[0].shape, output[1].shape, output[2].shape)