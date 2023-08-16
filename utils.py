def createConfusionMatrix(y_test,y_pred, plt_name):
    # No of classes for different datasets -> Trento - 6, MUUFL - 11, Houston - 15
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(6),range(6))
    df_cm.columns = ['Buildings','Woods', 'Roads', 'Apples', 'ground', 'Vineyard']
    # df_cm.columns = ["Grass-stressed","Tree","Water","Commercial","Highway","Parking-lot1","Tennis-court","Grass-healthy","Grass-synthetic","Soil","Residential","Road","Railway","Parking-lot2","Running-track"]
    df_cm = df_cm.rename({0:'Buildings',1:'Woods', 2:'Roads', 3:'Apples', 4:'ground', 5:'Vineyard'})
    # df_cm = df_cm.rename({0:"Grass-stressed",1:"Tree",2:"Water",3:"Commercial",4:"Highway",5:"Parking-lot1",6:"Tennis-court",7:"Grass-healthy",8:"Grass-synthetic",9:"Soil",10:"Residential",11:"Road",12:"Railway",13:"Parking-lot2",14:"Running-track"})
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=0.9)
    plt.figure(figsize=(30,30))
    sns.heatmap(df_cm, cmap="Blues",annot=True,annot_kws={"size": 16}, fmt='g')
    plt.savefig('Cross-HL_'+str(plt_name)+'.eps', format='eps')

def AvgAcc_andEachClassAcc(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    class_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(class_acc)
    return class_acc, average_acc

def result_reports(xtest,xtest2,ytest,name,model, iternum):
    y_pred = np.empty((len(ytest)), dtype=np.float32)
    number = len(ytest) // test_batch_size
    for i in range(number):
        temp = xtest[i * test_batch_size:(i + 1) * test_batch_size, :, :]
        temp = temp.cuda()
        temp1 = xtest2[i * test_batch_size:(i + 1) * test_batch_size, :, :]
        temp1 = temp1.cuda()
        temp2 = model(temp,temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        y_pred[i * test_batch_size:(i + 1) * test_batch_size] = temp3.cpu()
        del temp, temp2, temp3,temp1

    if (i + 1) * test_batch_size < len(ytest):
        temp = xtest[(i + 1) * test_batch_size:len(ytest), :, :]
        temp = temp.cuda()
        temp1 = xtest2[(i + 1) * test_batch_size:len(ytest), :, :]
        temp1 = temp1.cuda()
        temp2 = model(temp,temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        y_pred[(i + 1) * test_batch_size:len(ytest)] = temp3.cpu()
        del temp, temp2, temp3,temp1

    y_pred = torch.from_numpy(y_pred).long()

    overall_acc = accuracy_score(ytest, y_pred)
    confusion_mat = confusion_matrix(ytest, y_pred)
    class_acc, avg_acc = AvgAcc_andEachClassAcc(confusion_mat)
    kappa_score = cohen_kappa_score(ytest, y_pred)
    createConfusionMatrix(ytest, y_pred, str(name)+'_test_'+str(iternum)+'')

    return confusion_mat, overall_acc*100, class_acc*100, avg_acc*100, kappa_score*100