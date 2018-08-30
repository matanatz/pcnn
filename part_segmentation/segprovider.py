import os
import sys
import numpy as np
import h5py
BASE_DIR = './'
sys.path.append(BASE_DIR)
from provider import Provider


class SegmentationProvider(Provider):
    def __init__(self):
        DATA_DIR = os.path.join(BASE_DIR,'segmentation_data')

        self.train_files = os.path.join(os.path.join(DATA_DIR, 'hdf5_data'),'train_hdf5_file_list.txt')
        self.val_files = os.path.join(os.path.join(DATA_DIR, 'hdf5_data'),'val_hdf5_file_list.txt')

    def getValDataFiles(self):
        return self.getDataFiles(self.val_files)

    def getTrainDataFiles(self):
        return self.getDataFiles(self.train_files)

    def getDataFiles(self,list_filename):
        return [line.rstrip() for line in open(list_filename)]

    def load_h5(self,h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return (data, label)

    def loadDataFile(self,filename):
        return self.load_h5(os.path.join(os.path.join(BASE_DIR,'hdf5_data')),filename)
        #verts,_ = read_off(filename)
        #return verts

    def read_off(file):
        if 'OFF' != file.readline().strip():
            raise ('Not a valid OFF header')
        n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = []
        for i_vert in range(n_verts):
            verts.append([float(s) for s in file.readline().strip().split(' ')])
        faces = []
        for i_face in range(n_faces):
            faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
        return verts, faces

    def load_h5_data_label_seg(self,h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:].astype(np.int32)
        seg = f['pid'][:]
        return (data, label, seg)


    def loadDataFile_with_seg(self,filename):
        return self.load_h5_data_label_seg(filename)
