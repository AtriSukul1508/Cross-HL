import sys
sys.path.append("./../")
import os
import numpy as np
import random
from torch import einsum
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as dataf
from torch.utils.data import Dataset
from scipy import io
from scipy.io import loadmat as loadmat
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import time
from PIL import Image
import math
from operator import truediv
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from torchsummary import summary
import matplotlib.pyplot as plt
import logger
import torch.backends.cudnn as cudnn
import re
from pathlib import Path
import copy
cudnn.deterministic = True
cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"]="0"
datasetNames = ["Trento"] # ["Trento", "MUUFL", "Houston"]
MultiModalData = 'LiDAR'
modelName = 'Cross-HL'

patchsize = 11
batch_size = 64 # batch size for training 
test_batch_size = 500
EPOCHS = 100
learning_rate = 5e-4
FM = 16
FileName = 'CrossHL'
num_heads = 8 # d_h = number of mhsa heads
mlp_dim = 512
depth = 2 # Number of transformer encoder layer
num_iterations = 3
train_loss = []

def seed_val(seed=14):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


for dataset in datasetNames:
        print(f"---------------------------------- Details for {dataset} dataset---------------------------------------------")
        print('\n')
        try:
            os.makedirs(dataset)
        except FileExistsError:
            pass

        train_dataset = HSI_LiDAR_DatasetTrain(dataset=dataset)
        test_dataset = HSI_LiDAR_DatasetTest(dataset=dataset)
        print(len(train_dataset),len(test_dataset))
        NC = train_dataset.hs_ims.shape[1]

        NCLidar = train_dataset.lid_ims.shape[1]
        Classes = len(torch.unique(train_dataset.lbs))

        train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers= 4)


        test_patch_hsi = test_dataset.hs_ims
        test_patch_lidar = test_dataset.lid_ims
        test_label = test_dataset.lbs

        KAPPA = []
        OA = []
        AA = []
        ELEMENT_ACC = np.zeros((num_iterations, Classes)) # 3xNC

        seed_val(14)
        for iterNum in range(num_iterations):
            print('\n')
            print("---------------------------------- Summary ---------------------------------------------")
            print('\n')
            model = FuseFormer(FM=FM, NC=NC, NCLidar=NCLidar, Classes=Classes,patchsize = patchsize).cuda()
            summary(model, [(NC, patchsize**2),(NCLidar,patchsize**2)])

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5e-3)
            loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
            BestAcc = 0
            torch.cuda.synchronize()
            print('\n')
            print(f"---------------------------------- Training started for {dataset} dataset ---------------------------------------------")
            print('\n')
            start = time.time()
            # train and test the proposed model
            for epoch in range(EPOCHS):
                for step, (batch_hsi, batch_ldr, batch_lbl) in enumerate(train_loader):
                    # print(f"Step:{step} batch_hsi shape :{batch_hsi.shape} batch_ldr shape : {batch_ldr.shape} batch_lbl shape: {batch_lbl.shape}")
                    batch_hsi = batch_hsi.cuda()
                    batch_ldr = batch_ldr.cuda()
                    batch_lbl = batch_lbl.cuda()
                    out= model(batch_hsi, batch_ldr)
                    loss = loss_func(out, batch_lbl)
                    optimizer.zero_grad()  # Clearing gradients
                    loss.backward()  
                    optimizer.step()

                    if step % 50 == 0:
                        model.eval()
                        y_pred = np.empty((len(test_label)), dtype='float32')
                        number = len(test_label) // test_batch_size
                        for i in range(number):
                            temp = test_patch_hsi[i * test_batch_size:(i + 1) * test_batch_size, :, :]
                            temp = temp.cuda()
                            temp1 = test_patch_lidar[i * test_batch_size:(i + 1) * test_batch_size, :, :]
                            temp1 = temp1.cuda()
                            temp2 = model(temp, temp1)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            y_pred[i * test_batch_size:(i + 1) * test_batch_size] = temp3.cpu()
                            del temp, temp1, temp2, temp3

                        if (i + 1) * test_batch_size < len(test_label):
                            temp = test_patch_hsi[(i + 1) * test_batch_size:len(test_label), :, :]
                            temp = temp.cuda()
                            temp1 = test_patch_lidar[(i + 1) * test_batch_size:len(test_label), :, :]
                            temp1 = temp1.cuda()
                            temp2 = model(temp, temp1)
                            temp3 = torch.max(temp2, 1)[1].squeeze()
                            y_pred[(i + 1) * test_batch_size:len(test_label)] = temp3.cpu()
                            del temp, temp1, temp2, temp3

                        y_pred = torch.from_numpy(y_pred).long()
                        accuracy = torch.sum(y_pred == test_label).type(torch.FloatTensor) / test_label.size(0)

                        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % (accuracy*100))
                        train_loss.append(loss.data.cpu().numpy())

                        if accuracy > BestAcc:

                            BestAcc = accuracy

                            torch.save(model.state_dict(), dataset+'/net_params_'+FileName+'.pkl')


                        model.train()
                scheduler.step()
            torch.cuda.synchronize()
            end = time.time()
            print('\nThe train time (in seconds) is:', end - start)
            Train_time = end - start


            model.load_state_dict(torch.load(dataset+'/net_params_'+FileName+'.pkl'))

            model.eval()

            confusion_mat, overall_acc, class_acc, avg_acc, kappa_score = result_reports(test_patch_hsi,test_patch_lidar,test_label,dataset,model, iterNum)
            KAPPA.append(kappa_score)
            OA.append(overall_acc)
            AA.append(avg_acc)
            ELEMENT_ACC[iterNum, :] = class_acc
            torch.save(model, dataset+'/best_model_'+FileName+'_Iter'+str(iterNum)+'.pt')
            print('\n')
            print("Overall Accuracy = ", overall_acc)
            print('\n')
        print(f"---------- Training Finished for {dataset} dataset -----------")
        print("\nThe Confusion Matrix")
        logger.log_result(OA, AA, KAPPA, ELEMENT_ACC,'./' + dataset +'/'+FileName+'_Report_' + dataset +'.txt')
