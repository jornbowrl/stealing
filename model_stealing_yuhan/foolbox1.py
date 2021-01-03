#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import foolbox as fb
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
from PIL import Image
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[12]:
class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    #model = models.resnet18(pretrained=True).eval()
    #model=torch.load('/data1/zyh/copycat/Framework/cifar_model.pth')

    model =AlexNet()
    path = "./cifar_net.pth"
    #path = '/data1/zyh/copycat/Framework/cifar_model.pth'
    #model.load_state_dict(torch.load('/data1/zyh/copycat/Framework/cifar_model.pth'))
    #pretrained_dict = {k: v for k, v in model_pretrained.items() if k in model_dict}
    #model_dict.update(pretrained_dict)
    #model.load_state_dict(state_dict)
    model.load_state_dict(torch.load(path),strict=True)
    model.eval()

    print(type(model))
    #preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    preprocessing = dict(mean=[0.5]*3, std=[0.5]*3, axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)


    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    #test_dataset = torchvision.datasets.CIFAR10(root='~/.torch/',
    #                                         train=True,
    #                                         #transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]),
    #                                         transform = transforms.Compose([transforms.ToTensor()]),
    #                                         download=True)
    #test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                       batch_size=128, #该参数表示每次读取的批样本个数
    #                                       shuffle=False) #该参数表示读取时是否打乱样本顺序
    #                                       # 创建迭代器
    #data_iter = iter(test_loader)

    #images, labels = next(data_iter)
    # 当迭代开始时, 队列和线程开始读取数据
    #images, labels = data_iter.next()
    #images=images.to(device)
    #labels=labels.to(device)
    #im=images
    #images=im.resize(100,3,128,128)
    images, labels = ep.astensors(*samples(fmodel, dataset="cifar10", batchsize=16))
    #images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    #print(images.shape)
    clean_acc = accuracy(fmodel, images, labels)
    
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    # apply the attack
    attack = LinfPGD()
    '''epsilons = [
        0.0,
        0.0002,
        0.0005,
        0.0008,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.01,
        0.1,
        0.3,
        0.5,
        1.0,
    ]'''
    epsilons = [
        0.0005,
        0.001,
        0.002,
        0.01,
        0.1,
    ]
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    print(type(raw_advs))
    print("atest")
    # calculate and report the robust accuracy (the accuracy of the model when
    # it is attacked)
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # we can also manually check this
    # we will use the clipped advs instead of the raw advs, otherwise
    # we would need to check if the perturbation sizes are actually
    # within the specified epsilon bound
    print()
    print("we can also manually check this:")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break
    fig = plt.gcf()
    os.makedirs("./image/",exist_ok=True)
    for i in range(len(raw_advs)):
        img_v = raw_advs[i].raw
        torchvision.utils.save_image(img_v, './image/'+str(i) +'.png')
#         plt.imsave('./image/'+str(i) +'.png',raw_advs[i].numpy())
#         i += 1
#         plt.close(fig)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




