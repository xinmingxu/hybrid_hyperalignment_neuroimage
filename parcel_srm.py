import os, itertools, pickle
import numpy as np
from scipy import stats
import scipy.spatial.distance as sp_distance
from sklearn.svm import NuSVC
import nibabel as nib
from nibabel import freesurfer as nfs

from brainiak.isc import isc
from brainiak.fcma.util import compute_correlation
import brainiak.funcalign.srm
from brainiak import image, io

lh = nfs.read_annot('/dartfs-hpc/scratch/ps16421/haxby_class/lh.Schaefer2018_700Parcels_17Networks_order.annot')
rh = nfs.read_annot('/dartfs-hpc/scratch/ps16421/haxby_class/rh.Schaefer2018_700Parcels_17Networks_order.annot')
bh = np.concatenate((lh[0], rh[0]+350))

all_runs = np.arange(1, 5)
train_run_combos = list(itertools.combinations(all_runs, 3))

for train in train_run_combos: 
    test = np.setdiff1d(all_runs, train)
    print('training on runs {r}; testing on run {n}'.format(r=train, n=test))

    train_data_stacked = np.load(f'/dartfs-hpc/scratch/ps16421/haxby_class/whiplash_parcel/dss_train_{test[0]}.npy')
    test_data_stacked = np.load(f'/dartfs-hpc/scratch/ps16421/haxby_class/whiplash_parcel/dss_test_{test[0]}.npy')

    train_data = [train_data_stacked[i].T for i in range(train_data_stacked.shape[0])]
    test_data = [test_data_stacked[i].T for i in range(test_data_stacked.shape[0])]
    shared_test = []
    w = []

    for p in (list(range(1,350)) + list(range(351,700))):
    # for p in (list(range(1,3))):
        print(f"now parcel {p}")
        mask = bh == p
        mask_size = mask.sum()
        train_data_parcel = [train_data[sub][mask, :] for sub in range(len(train_data))]
        test_data_parcel = [test_data[sub][mask, :] for sub in range(len(train_data))]

        features = 20 if mask_size>20 else mask_size # How many features will you fit?
        n_iter = 20  # How many iterations of fitting will you perform
        srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=features)
        srm.fit(train_data_parcel)
        shared_test_parcel = srm.transform(test_data_parcel)

        # Zscore the transformed test data
        for subject in range(len(train_data)):
            shared_test_parcel[subject] = stats.zscore(shared_test_parcel[subject], axis=1, ddof=1)

        shared_test.append(np.stack(shared_test_parcel))
        w.append(np.stack(srm.w_))

        with open('/dartfs-hpc/scratch/ps16421/haxby_class/whiplash_parcel/shared_test_{test[0]}.pkl', 'wb') as f:
            pickle.dump(shared_test, f)
        with open('/dartfs-hpc/scratch/ps16421/haxby_class/whiplash_parcel/w_{test[0]}.pkl', 'wb') as f:
            pickle.dump(w, f)