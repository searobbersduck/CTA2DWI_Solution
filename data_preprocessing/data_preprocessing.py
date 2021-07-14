#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_preprocessing.py
@Time    :   2021/07/13 09:54:03
@Author  :   searobbersandduck 
@Version :   1.0
@Contact :   searobbersandduck@gmail.com
@License :   (C)Copyright 2020-2021, MIT
@Desc    :   None
'''

# here put the import lib

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(ROOT)

from tqdm import tqdm
from glob import glob
import shutil
import SimpleITK as sitk

from external_lib.MedCommon.utils.data_io_utils import DataIO


def step_1_check_folder_format(root, out_root):
    '''
    root = '/data/medical/brain/gan/hospital_6_multi_classified/CTA2DWI-多中心-20201102'
    out_root = '/data/medical/brain/gan/cta2dwi_multi_classified'


    note: 
        1. 确认各组数据中是否都至少包含CTA和DWI两组数据，并统计各自数据的数量
        2. 将数据重新copy到新的路径下，并只保留CTA和DWI两个文件夹，
            同时有CTA1和CTA2两个文件夹时，将CTA1重命名成CTA

    .
    ├── CTA阴性（108例）
    │   ├── 六院-CTA阴性（69）
    │   ├── 南通大学-阴性-血管(14)
    │   └── 闵中心-阴性-血管（25）
    └── 阳性-闭塞(188例）
        ├── 六院-DWI闭塞病例(105)
        ├── 六院-阳性-血管闭塞（25）
        ├── 六院-阳性-血管闭塞（37）
        ├── 南通大学-阳性-血管闭塞（5）
        └── 闵中心-阳性-血管闭塞（16）

    具体展开其中的一个文件夹：
    闵中心-阳性-血管闭塞（16）$ tree -L 2
    .
    ├── 101878640
    │   ├── CTA
    │   └── DWI
    ├── 102512839-101477685
    │   ├── CTA
    │   └── DWI
    ├── 102661445
    │   ├── CTA
    │   └── DWI
    ├── 102869917
    │   ├── CTA
    │   └── DWI

    '''
    pn_roots = os.listdir(root)
    '''
    .
    ├── CTA阴性（108例）
    └── 阳性-闭塞(188例）    
    '''

    def comprass_modalities(modalities):
        pairs = []
        if 'CTA' in modalities:
            pairs.append('CTA')
        elif 'CTA1' in modalities:
            pairs.append('CTA1')
        elif 'CTA2' in modalities:
            pairs.append('CTA2')
        
        if 'DWI' in modalities:
            pairs.append('DWI')
        elif 'DWI1' in modalities:
            pairs.append('DWI1')
        elif 'DWI2' in modalities:
            pairs.append('DWI2')
        return pairs


    for pn_root in pn_roots:
        pn_path = os.path.join(root, pn_root)
        for hospital_name in os.listdir(pn_path):
            hospital_path = os.path.join(pn_path, hospital_name)
            
            for pid in tqdm(os.listdir(hospital_path)):
                try:
                    if len(pid) != 7:
                        continue
                    pid_path = os.path.join(hospital_path, pid)
                    modalities = os.listdir(pid_path)
                    pairs_modalities = []
                    pairs_modalities = comprass_modalities(modalities)
                    
                    for m in pairs_modalities:
                        src_path = os.path.join(pid_path, m)
                        dst_path = src_path.replace(root, out_root)
                        if dst_path.endswith('1') or dst_path.endswith('2'):
                            dst_path = dst_path[:-1]
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copytree(src_path, dst_path)
                        print('copy from {} to {o}'.format(src_path, dst_path))

                    
                    if len(pairs_modalities) != 2:
                        print(pid_path)
                except Error as e:
                    print('Error case:\t', pid)

def convert_dcm_to_nii(in_root, out_root):
    '''
    tree -L 1
    .
    ├── 1124013
    ├── 1140092
    ├── 1195207
    ├── 1399063
    ├── 1424031
    ├── 1534457
    ├── 1870593
    ├── 1944927

    '''
    
    for pid in tqdm(os.listdir(in_root)):
        patient_path = os.path.join(in_root, pid)
        out_sub_root = os.path.join(out_root, pid)
        os.makedirs(out_sub_root, exist_ok=True)
        cta_path = os.path.join(patient_path, 'CTA')
        cta_image = DataIO.load_dicom_series(cta_path)
        out_cta_file = os.path.join(out_sub_root, 'CTA.nii.gz')
        sitk.WriteImage(cta_image['sitk_image'], out_cta_file)

        dwi_path = os.path.join(patient_path, 'DWI')
        dwi_image = DataIO.load_dicom_series(dwi_path)
        out_dwi_file = os.path.join(out_sub_root, 'DWI.nii.gz')
        sitk.WriteImage(dwi_image['sitk_image'], out_dwi_file)

def step_2_dcm_to_nii(in_root, out_root):
    for pn_root in os.listdir(in_root):
        pn_path = os.path.join(in_root, pn_root)
        for hospital_name in os.listdir(pn_path):
            hospital_path = os.path.join(pn_path, hospital_name)
            convert_dcm_to_nii(hospital_path, out_root)


def cerebral_parenchyma_segmentation_new_algo(
        data_root=None, 
        out_dir = None
    ):
    import torch
    from external_lib.MedCommon.experiments.seg.brain.parenchyma.inference.inference import load_inference_opts
    from external_lib.MedCommon.segmentation.runner.train_seg import SegmentationTrainer
    opts = load_inference_opts()
    model = SegmentationTrainer.load_model(opts)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    for pid in tqdm(os.listdir(data_root)):
        pid_path = os.path.join(data_root, pid)
        if not os.path.isdir(pid_path):
            print('patient path not exist!\t{}'.format(pid_path))
            continue
        cta_file = os.path.join(pid_path, 'CTA.nii.gz')
        if not os.path.isfile(cta_file):
            print('cta file not exist!\t{}'.format(cta_file))
            continue
        image, pred_mask = SegmentationTrainer.inference_one_case(model, cta_file, is_dcm=False)
        out_cta_dir = os.path.join(out_dir, pid, 'CTA')
        os.makedirs(out_cta_dir, exist_ok=True)
        out_cta_file = os.path.join(out_cta_dir, 'CTA.nii.gz')
        out_cta_mask_file = os.path.join(out_cta_dir, 'CTA_MASK.nii.gz')

        sitk.WriteImage(image, out_cta_file)
        sitk.WriteImage(pred_mask, out_cta_mask_file)  


def data_preprocessing():
    data_root = '/data/medical/brain/gan/cta2dwi_multi_classified'

    # step_2_dcm_to_nii(os.path.join(data_root, '0.ori'), 
    #     os.path.join(data_root, '3.sorted_nii'))

    cerebral_parenchyma_segmentation_new_algo(
        os.path.join(data_root, '3.sorted_nii'), 
        os.path.join(data_root, '3.sorted_mask')
    )


if __name__ == '__main__':
    # step_1_check_folder_format('/data/medical/brain/gan/hospital_6_multi_classified/CTA2DWI-多中心-20201102', 
    #     '/data/medical/brain/gan/cta2dwi_multi_classified')

    data_preprocessing()