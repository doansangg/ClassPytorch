import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix
import sys
from operator import truediv
import pandas as pd
from dataloader import get_loader_test
import seaborn as sns
import matplotlib.pyplot as plt
# Paths for image directory and model
# EVAL_DIR='eval_ds'
EVAL_MODEL='models/model.pth'
def get_f1_score(confusion_matrix, i):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for j in range(len(confusion_matrix)):
        if (i == j):
            TP += confusion_matrix[i, j]
            tmp = np.delete(confusion_matrix, i, 0)
            tmp = np.delete(tmp, j, 1)

            TN += np.sum(tmp)
        else:
            if (confusion_matrix[i, j] != 0):

                FN += confusion_matrix[i, j]
            if (confusion_matrix[j, i] != 0):

                FP += confusion_matrix[j, i]

    recall = TP / (FN + TP)
    precision = TP / (TP + FP)
    f1_score = 2 * 1/(1/recall + 1/precision)

    return precision,recall,f1_score
# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and nuber of cpu's
num_cpu = multiprocessing.cpu_count()
bs = 8

# # Prepare the eval data loader
# eval_transform=transforms.Compose([
#         transforms.Resize(size=256),
#         transforms.CenterCrop(size=224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])])

# eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
# eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
#                             num_workers=num_cpu, pin_memory=True)
eval_loader = get_loader_test('path_data/test.txt',64)
# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
len_test=open('path_data/test.txt','r')
len_test1=len(len_test.readlines())
#num_classes=len(eval_dataset.classes)
dsize=len_test1

# Class label names
class_names={0:'1 Hau hong',
1:'10 Ta trang',
2:'2 Thuc quan',
3:'3 Tam vi',
4:'4 Than vi',
5:'5 Phinh vi',
6:'6 Hang vi',
7:'7 Bo cong lon'
,8:'8 Bo cong nho'
,9:'9 Hanh ta trang'}
# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
############################################



# Evaluate the model accuracy on the dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy=100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, 
    overall_accuracy))

# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
 
index =['1 Hau hong','10 Ta trang','2 Thuc quan','3 Tam vi','4 Than vi','5 Phinh vi','6 Hang vi','7 Bo cong lon','8 Bo cong nho','9 Hanh ta trang'] 
columns = ['1 Hau hong','10 Ta trang','2 Thuc quan','3 Tam vi','4 Than vi','5 Phinh vi','6 Hang vi','7 Bo cong lon','8 Bo cong nho','9 Hanh ta trang'] 
cm_df = pd.DataFrame(conf_mat,columns,index)   
print(cm_df)                   
plt.figure(figsize=(10,6))  
sns_plot=sns.heatmap(cm_df, annot=True)
sns_plot.figure.savefig("output.png")
print('Confusion Matrix')
print('-'*16)
print(type(conf_mat),'\n')
print(conf_mat,'\n')
# tp = np.diag(conf_mat)
# prec = list(map(truediv, tp, np.sum(conf_mat, axis=0)))
# rec = list(map(truediv, tp, np.sum(conf_mat, axis=1)))
# print ('Precision: {}\nRecall: {}'.format(prec, rec))
column1 = []
num_classes = 10
for i in range(num_classes):
    precision,recall,f1_score=get_f1_score(conf_mat,i)
    column1.append([class_names[i],precision,recall,f1_score])
    #print(column)
a=pd.DataFrame(column1,columns=['label','precision','recall','f1-score'])

print(a)
# Per-class accuracy
# class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
# print('Per class accuracy')
# print('-'*18)
# for label,accuracy in zip(eval_dataset.classes, class_accuracy):
#      class_name=class_names[int(label)]
#      print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))

'''
Sample run: python eval.py eval_ds
'''
