class MultiTaskTrainer:
    """多任务训练器"""
    def __init__(self, model, dataset, device='cuda'):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.dataloader:
            images = batch['img'].to(self.device)
            targets = {
                'boxes': batch['bboxes'].to(self.device),
                'cls': batch['cls'].to(selfdevice),
                'mask': batch['mask'].to(self.device)
            }
            
            # 前向传播
            outputs = self.model(images)
            
            # 多任务损失计算
            loss = self.compute_loss(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.dataloader)

    def compute_loss(self, outputs, targets):
        """综合检测和分割损失"""
        det_loss = F.binary_cross_entropy(outputs['det'], targets['cls'])
        seg_loss = F.dice_loss(outputs['mask'], targets['mask'])
        return det_loss + 0.8*seg_loss