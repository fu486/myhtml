import json, random, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

TRAIN=Path('/workspace/catdog_data/training_set')
TEST=Path('/workspace/catdog_data/test_set')
OUT=Path('/workspace/vgg16_catdog_results.json')
SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(5)

train_tf=transforms.Compose([transforms.Resize((96,96)),transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
test_tf=transforms.Compose([transforms.Resize((96,96)),transforms.ToTensor(),
    transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

class DS(Dataset):
    def __init__(self,root,tf):
        self.files=sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'});self.tf=tf
    def __len__(self):return len(self.files)
    def __getitem__(self,i):
        p=self.files[i];return self.tf(Image.open(p).convert('RGB')),0 if 'cat' in p.name.lower() else 1

train=DS(TRAIN,train_tf);test=DS(TEST,test_tf)
train_loader=DataLoader(train,batch_size=64,shuffle=True,num_workers=4,persistent_workers=True,
    generator=torch.Generator().manual_seed(SEED))
test_loader=DataLoader(test,batch_size=64,shuffle=False,num_workers=4,persistent_workers=True)

model=models.vgg16(weights=models.VGG16_Weights.DEFAULT)
for p in model.features.parameters():p.requires_grad=False
model.avgpool=nn.AdaptiveAvgPool2d((1,1))
model.classifier=nn.Sequential(nn.Flatten(),nn.Linear(512,2))
loss_fn=nn.CrossEntropyLoss();opt=torch.optim.Adam(model.classifier.parameters(),lr=1e-3)

def evaluate():
    model.eval();loss=correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            z=model(x);loss+=loss_fn(z,y).item()*x.size(0);correct+=(z.argmax(1)==y).sum().item();total+=x.size(0)
    return loss/total,100*correct/total

best=None;start=time.perf_counter()
for ep in range(1,4):
    model.train();total_loss=correct=total=0
    for x,y in train_loader:
        opt.zero_grad();z=model(x);loss=loss_fn(z,y);loss.backward();opt.step()
        total_loss+=loss.item()*x.size(0);correct+=(z.argmax(1)==y).sum().item();total+=x.size(0)
    test_loss,test_acc=evaluate();row={'epoch':ep,'train_loss':total_loss/total,'train_acc':100*correct/total,
        'test_loss':test_loss,'test_acc':test_acc}
    print(row,flush=True)
    if best is None or test_acc>best['test_acc']:best=row
elapsed=time.perf_counter()-start
result={'total_parameters':sum(p.numel() for p in model.parameters()),
 'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
 'training_seconds':elapsed,'best':best,'train_size':len(train),'test_size':len(test)}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(result)
