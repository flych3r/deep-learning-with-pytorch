import copy
import csv
import functools
import glob
import logging
import os
import random
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Text, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.functional as F
from cache import get_cache
from numpy.core.records import array
from torch.functional import Tensor
from torch.utils.data.dataset import Dataset

log = logging.getLogger(__name__)

data_path = Path('../data/04')
candidate_info_tuple = namedtuple(
    'candidate_info_tuple', 'is_nodule, diameter_mm, series_uid, center_xyz'
)
irc_tuple = namedtuple('irc_tuple', ['index', 'row', 'col'])
xyz_tuple = namedtuple('xyz_tuple', ['x', 'y', 'z'])


raw_cache = get_cache('luna')


@functools.lru_cache(1)
def get_candidate_info_list(
    require_on_disk: Optional[bool] = True
) -> List[candidate_info_tuple]:
    mhd_list = glob.glob(str(data_path / 'subset*/*.mhd'))
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = defaultdict(list)
    with open(data_path / 'annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            diameter_dict[series_uid].append(
                (annotation_center_xyz, annotation_diameter_mm)
            )

    candidate_info_list = []
    with open(data_path / 'candidates.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in present_on_disk_set and require_on_disk:
                continue
            is_nodule = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])
            candidate_diameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup
            for i in range(3):
                delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                if delta_mm > annotation_diameter_mm / 4:
                    break
                else:
                    candidate_diameter_mm = annotation_diameter_mm
                    break
            candidate_info_list.append(candidate_info_tuple(
                is_nodule,
                candidate_diameter_mm,
                series_uid,
                candidate_center_xyz,
            ))

    candidate_info_list.sort(reverse=True)
    return candidate_info_list


def irc2xyz(coord_irc, origin_xyz, vx_size_xyz, direction_a) -> xyz_tuple:
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vx_size_a = np.array(vx_size_xyz)
    coords_xyz = (direction_a @ (cri_a * vx_size_a)) + origin_a
    return xyz_tuple(*coords_xyz)


def xyz2irc(coord_xyz, origin_xyz, vx_size_xyz, direction_a) -> irc_tuple:
    origin_a = np.array(origin_xyz)
    vx_size_a = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vx_size_a
    cri_a = np.round(cri_a)
    return irc_tuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))


class CtScan:
    def __init__(self, series_uid: Text):
        mhd_path = glob.glob(
            str(data_path / 'subset*/{}.mhd'.format(series_uid))
        )[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_arr.clip(-1000, 1000, ct_arr)

        self.series_uid = series_uid
        self.hu_arr = ct_arr

        self.origin_xyz = xyz_tuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = xyz_tuple(*ct_mhd.GetSpacing())
        self.direction_arr = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(
        self,
        center_xyz: Tuple[float, float, float],
        width_irc: Tuple[float, float, float]
    ) -> Tuple[array, irc_tuple]:
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vx_size_xyz,
            self.direction_arr,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            if not center_val >= 0 and center_val < self.hu_arr.shape[axis]:
                raise ValueError(repr([
                    self.series_uid, center_xyz, self.origin_xyz,
                    self.vx_size_xyz, center_irc, axis
                ]))

            warn = 'Crop outside of CT array: {} {}, center:{} shape:{} width:{}'
            if start_ndx < 0:
                log.warning(
                    warn.format(
                        self.series_uid, center_xyz,
                        center_irc, self.hu_arr.shape, width_irc
                    )
                )
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_arr.shape[axis]:
                log.warning(
                    warn.format(
                        self.series_uid, center_xyz,
                        center_irc, self.hu_arr.shape, width_irc
                    )
                )
                end_ndx = self.hu_arr.shape[axis]
                start_ndx = int(self.hu_arr.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_arr[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def get_CtScan(series_uid: Text) -> CtScan:
    return CtScan(series_uid)


@raw_cache.memoize(typed=True)
def get_CtScan_raw_candidate(
    series_uid: Text,
    center_xyz: Tuple[float, float, float],
    width_irc: Tuple[float, float, float]
) -> Tuple[array, irc_tuple]:
    ct = get_CtScan(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc


def get_CtScan_augmented_candidate(
    augmentation_dict: Dict,
    series_uid: Text,
    center_xyz: Tuple[float, float, float],
    width_irc: Tuple[float, float, float],
    use_cache: Optional[bool] = True
) -> Tuple[array, irc_tuple]:
    if use_cache:
        ct_chunk, center_irc = get_CtScan_raw_candidate(
            series_uid, center_xyz, width_irc
        )
    else:
        ct = get_CtScan(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_tensor = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)
    transform_tensor = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_tensor[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_tensor[i, 3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_tensor[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * np.pi * 2
        s = torch.sin(angle_rad)
        c = torch.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_tensor @= rotation_t

    affine_tensor = F.affine_grid(
        transform_tensor[:3].unsqueeze(0).to(torch.float32),
        ct_tensor.size(),
        align_corners=False,
    )

    augmented_chunk = F.grid_sample(
        ct_tensor,
        affine_tensor,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_tensor = torch.randn_like(augmented_chunk)
        noise_tensor *= augmentation_dict['noise']
        augmented_chunk += noise_tensor

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(
        self,
        val_stride: Optional[int] = 0,
        is_val_set: Optional[bool] = None,
        series_uid: Optional[Text] = None,
        ratio: Optional[int] = 0,
        sort_by='random',
        augmentation_dict=None,
        candidate_info_list=None,
    ):
        super().__init__()
        self.ratio = ratio
        self.augmentation_dict = augmentation_dict

        if candidate_info_list:
            self.candidate_info_list = copy.copy(candidate_info_list)
            self.use_cache = False
        else:
            self.candidate_info_list = copy.copy(get_candidate_info_list())
            self.use_cache = True

        if series_uid:
            self.candidate_info_list = [
                x for x in self.candidate_info_list if x.series_uid == series_uid
            ]

        if is_val_set:
            if not val_stride > 0:
                raise ValueError(repr(val_stride))
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            if not self.candidate_info_list:
                raise ValueError('empty candidates')
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            if not self.candidate_info_list:
                raise ValueError('empty candidates')

        if sort_by == 'random':
            random.shuffle(self.candidate_info_list)
        elif sort_by == 'series_uid':
            self.candidate_info_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sort_by == 'label_and_size':
            pass
        else:
            raise Exception('Unknown sort: ' + repr(sort_by))

        self.neg_list = [
            nt for nt in self.candidate_info_list if nt.is_nodule
        ]
        self.pos_list = [
            nt for nt in self.candidate_info_list if nt.is_nodule
        ]

        log.info('{!r}: {} {} samples, {} neg, {} pos, {} ratio'.format(
            self,
            len(self.candidate_info_list),
            'validation' if is_val_set else 'training',
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio) if self.ratio else 'unbalanced'
        ))

    def shuffle_samples(self):
        if self.ratio:
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)

    def __len__(self) -> int:
        return len(self.candidate_info_list)

    def __getitem__(self, ndx: int) -> Tuple[Tensor, Tensor, Text, Tensor]:
        if self.ratio:
            pos_ndx = ndx // (self.ratio + 1)

            if ndx % (self.ratio + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidate_info_tup = self.neg_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidate_info_tup = self.pos_list[pos_ndx]
        else:
            candidate_info_tup = self.candidate_info_list[ndx]

        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_tensor, center_irc = get_CtScan_augmented_candidate(
                self.augmentation_dict,
                candidate_info_tup.series_uid,
                candidate_info_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_arr, center_irc = get_CtScan_raw_candidate(
                candidate_info_tup.series_uid,
                candidate_info_tup.center_xyz,
                width_irc,
            )

            candidate_tensor = torch.from_numpy(candidate_arr)
            candidate_tensor = candidate_tensor.to(torch.float32)
            candidate_tensor = candidate_tensor.unsqueeze(0)
        else:
            ct = get_CtScan(candidate_info_tup.series_uid)
            candidate_arr, center_irc = ct.get_CtScan_raw_candidate(
                candidate_info_tup.center_xyz,
                width_irc,
            )
            candidate_tensor = torch.from_numpy(candidate_arr).to(torch.float32)
            candidate_tensor = candidate_tensor.unsqueeze(0)

        pos_tensor = torch.tensor([
            not candidate_info_tup.is_nodule,
            candidate_info_tup.is_nodule
        ], dtype=torch.long)

        return (
            candidate_tensor,
            pos_tensor,
            candidate_info_tup.series_uid,
            torch.tensor(center_irc),
        )
