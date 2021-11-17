import os, itertools, pickle
import numpy as np
import whiplash_utils as utils
from mvpa2.suite import *
from scipy import stats
from nibabel import freesurfer as nfs
from drive_hyperalignment_cross_validation import add_node_indices

def add_node_indices(dss_train):
    for ds in dss_train:
        ds.fa['node_indices'] = np.arange(ds.shape[1], dtype=int)
    return dss_train

lh = nfs.read_annot('/dartfs-hpc/scratch/ps16421/haxby_class/lh.Schaefer2018_700Parcels_17Networks_order.annot')
rh = nfs.read_annot('/dartfs-hpc/scratch/ps16421/haxby_class/rh.Schaefer2018_700Parcels_17Networks_order.annot')
bh = np.concatenate((lh[0], rh[0]+350))

all_runs = np.arange(1, 5)
train_run_combos = list(itertools.combinations(all_runs, 3))

for train in train_run_combos: 
    test = np.setdiff1d(all_runs, train)
    print('training on runs {r}; testing on run {n}'.format(r=train, n=test))

    train_data_stacked = np.load('/dartfs-hpc/scratch/ps16421/haxby_class/whiplash_parcel/dss_train_{t}.npy'.format(t=test[0]))
    test_data_stacked = np.load('/dartfs-hpc/scratch/ps16421/haxby_class/whiplash_parcel/dss_test_{t}.npy'.format(t=test[0]))

    train_data_stacked[np.isnan(train_data_stacked)]=0 
    test_data_stacked[np.isnan(test_data_stacked)]=0 

    aligned_test = []
    for p in (list(range(1,350)) + list(range(351,700))):
    # for p in (list(range(1,3))):
 
        print("now parcel {}".format(p))
        mask = bh == p
        mask_size = mask.sum()

        dss_train = []
        for sub in range(train_data_stacked.shape[0]):
            sub_train_masked = train_data_stacked[sub, :, mask]
            ds = Dataset(sub_train_masked.T)
            dss_train.append(ds)

        dss_train = add_node_indices(dss_train)

        dss_test = []
        for sub in range(test_data_stacked.shape[0]):
            sub_test_masked = test_data_stacked[sub, :, mask]
            ds = Dataset(sub_test_masked.T)
            dss_test.append(ds)

        dss_test = add_node_indices(dss_test)

        hyper = Hyperalignment()
        hypmaps = hyper(dss_train)
        dss_aligned = [h.forward(sd) for h, sd in zip(hypmaps, dss_test)]
        _ = [zscore(ds, chunks_attr=None) for ds in dss_aligned]

        aligned_test.append(np.stack(dss_aligned))
    
    with open('/dartfs-hpc/scratch/ps16421/haxby_class/whiplash_parcel/aligned_test_{t}.pkl'.format(t=test[0]), 'wb') as f:
        pickle.dump(aligned_test, f)