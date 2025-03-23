class SegmentationWrapper(nn.Module):
    """适配自定义分割模型"""
    def __init__(self, model_cfg):
        super().__init__()
        self.model = SegmentationModel(model_cfg)
        
    def forward(self, x):
        outputs = self.model(x)
        return {
            'det': outputs['det'],
            'mask': outputs['mask']
        }