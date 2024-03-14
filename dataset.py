class HSI_LiDAR_DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, dataset='Trento'):

        HSI = loadmat(f'./{dataset}11x11/HSI_Tr.mat')
        LiDAR = loadmat(f'./{dataset}11x11/LIDAR_Tr.mat')
        label = loadmat(f'./{dataset}11x11/TrLabel.mat')

        self.hs_image = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lidar_image = (torch.from_numpy(LiDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lbls = ((torch.from_numpy(label['Data'])-1).long()).reshape(-1)

    def __len__(self):
        return self.hs_image.shape[0]

    def __getitem__(self, i):
        return self.hs_image[i], self.lidar_image[i], self.lbls[i]

class HSI_LiDAR_DatasetTest(torch.utils.data.Dataset):
    def __init__(self, dataset='Trento'):

        HSI = loadmat(f'./{dataset}11x11/HSI_Te.mat')
        LiDAR = loadmat(f'./{dataset}11x11/LIDAR_Te.mat')
        label = loadmat(f'./{dataset}11x11/TeLabel.mat')

        self.hs_image = (torch.from_numpy(HSI['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lidar_image = (torch.from_numpy(LiDAR['Data'].astype(np.float32)).to(torch.float32)).permute(0,3,1,2)
        self.lbls = ((torch.from_numpy(label['Data'])-1).long()).reshape(-1)


    def __len__(self):
        return self.hs_image.shape[0]

    def __getitem__(self, i):
        return self.hs_image[i], self.lidar_image[i], self.lbls[i]
