import logging
from typing import List, Text

import matplotlib.pyplot as plt
from dataset import CtScan, LunaDataset, candidate_info_tuple

log = logging.getLogger(__name__)

c_lim = (-1000.0, 300)


def find_positive_samples(limit: int = 100) -> List[candidate_info_tuple]:
    ds = LunaDataset()

    positive_sample_list = []
    for sample_tup in ds.candidate_info_list:
        if sample_tup.is_nodule:
            print(len(positive_sample_list), sample_tup)
            positive_sample_list.append(sample_tup)

        if len(positive_sample_list) >= limit:
            break

    return positive_sample_list


def show_candidate(series_uid: Text, batch_ndx: int = None, **kwargs):
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    pos_list = [i for i, x in enumerate(ds.candidate_info_list) if x.is_nodule]

    if batch_ndx is None:
        if pos_list:
            batch_ndx = pos_list[0]
        else:
            log.warn('Warning: no positive samples found; using first negative sample.')
            batch_ndx = 0

    ct = CtScan(series_uid)
    ct_t, pos_t, series_uid, center_irc = ds[batch_ndx]
    ct_arr = ct_t[0].numpy()

    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_arr[int(center_irc[0])], clim=c_lim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_arr[:,int(center_irc[1])], clim=c_lim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_arr[:,:,int(center_irc[2])], clim=c_lim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc[0])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_arr[ct_arr.shape[0]//2], clim=c_lim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc[1])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_arr[:,ct_arr.shape[1]//2], clim=c_lim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc[2])), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_arr[:,:,ct_arr.shape[2]//2], clim=c_lim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_arr[index], clim=c_lim, cmap='gray')

    plt.show()
    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)
