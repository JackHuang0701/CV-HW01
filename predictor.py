class FloorPredictor:
    """定制预测器"""
    def __init__(self, model, session):
        self.model = model
        self.session = session
        
    def predict(self, img_path):
        img = self.preprocess(img_path)
        with torch.no_grad():
            outputs = self.model(img)
        return self.postprocess(outputs)
    
    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        return F.resize(img, (640,640)) / 255
    
    def postprocess(self, outputs):
        # 应用NMS并生成最终结果
        detections = non_max_suppression(outputs['det'])
        masks = F.sigmoid(outputs['mask']) > 0.5
        return {'boxes': detections, 'masks': masks}