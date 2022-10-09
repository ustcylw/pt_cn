import torch
from torchvision.models import AlexNet
from torchviz import make_dot
 
x=torch.rand(8,3,256,512)
model=AlexNet()
y=model(x)
 
# 调用make_dot()函数构造图对象
g = make_dot(y)
 
# 保存模型，以PDF格式保存
g.render('./Alex_model', view=False)
