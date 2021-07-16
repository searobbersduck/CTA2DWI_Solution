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
import numpy as np
import shutil

from external_lib.MedCommon.utils.data_io_utils import DataIO
from external_lib.MedCommon.utils.image_postprocessing_utils import ImagePostProcessingUtils
from external_lib.MedCommon.utils.mask_bounding_utils import MaskBoundingUtils
from external_lib.MedCommon.utils.mask_utils import MaskUtils


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

def step_3_3_segment_cerebral_parenchyma_connected_region(root_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'):
    # root_dir = '/data/medical/cardiac/cta2mbf/20201216/3.sorted_mask'
    for pid in tqdm(os.listdir(root_dir)):
        pid_path = os.path.join(root_dir, pid)
        if not os.path.isdir(pid_path):
            continue
        cta_root = os.path.join(pid_path, 'CTA')
        
        in_cta_file = os.path.join(cta_root, 'CTA_MASK.nii.gz')
        out_cta_file = os.path.join(cta_root, 'CTA_MASK_connected.nii.gz')

        try:
            if os.path.isfile(in_cta_file):
                in_mask = sitk.ReadImage(in_cta_file)
                out_mask_sitk = ImagePostProcessingUtils.get_maximal_connected_region_multilabel(in_mask, mask_labels=[1])
                out_mask_sitk = MaskUtils.fill_hole(out_mask_sitk, radius=4)
                sitk.WriteImage(out_mask_sitk, out_cta_file)
        except Exception as e:
            print(e)
            print('====> Error case:\t{}'.format(pid))

def extract_cta_cerebral_parenchyma_zlayers(
        cta_root, 
        mask_root, 
        out_root,
        cta_pattern = 'CTA/CTA.nii.gz', 
        mask_pattern = 'CTA/CTA_MASK_connected.nii.gz'):
    pids = os.listdir(mask_root)
    for pid in tqdm(pids):
        cta_file = os.path.join(cta_root, pid, cta_pattern)
        mask_file = os.path.join(mask_root, pid, mask_pattern)
        in_image = sitk.ReadImage(cta_file)
        in_mask = sitk.ReadImage(mask_file)
        out_image, out_mask = MaskBoundingUtils.extract_target_area_by_mask_zboundary(in_image, in_mask)
        out_dir = os.path.join(out_root, pid)
        os.makedirs(out_dir, exist_ok=True)
        out_image_file = os.path.join(out_dir, 'CTA.nii.gz')
        sitk.WriteImage(out_image, out_image_file)
        out_mask_file = os.path.join(out_dir, 'MASK.nii.gz')
        sitk.WriteImage(out_mask, out_mask_file)

def generate_dwi_bbox_mask(in_root, out_root, dwi_pattern='DWI.nii.gz', out_dwi_mask_pattern='DWI_BBOX_MASK.nii.gz'):
    for pid in tqdm(os.listdir(in_root)):
        dwi_file = os.path.join(in_root, pid, dwi_pattern)
        dwi_image = sitk.ReadImage(dwi_file)
        size = dwi_image.GetSize()
        size = size[::-1]
        bbox_mask_arr = np.ones(size, dtype=np.uint8)
        bbox_mask = sitk.GetImageFromArray(bbox_mask_arr)
        bbox_mask.CopyInformation(dwi_image)
        out_sub_dir = os.path.join(out_root, pid)
        os.makedirs(out_sub_dir, exist_ok=True)
        out_mask_file = os.path.join(out_sub_dir, out_dwi_mask_pattern)
        sitk.WriteImage(bbox_mask, out_mask_file)
        # copy dwi文件到指定路径，方便后续操作
        src_file = dwi_file
        dst_file = os.path.join(out_sub_dir, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)
        print('hello world!')

def extract_dwi_cerebral_parenchyma(
        dwi_root, 
        mask_root, 
        out_root, 
        dwi_pattern = 'registried_dwi.nii.gz',
        mask_pattern = 'MASK.nii.gz',
        out_dwi_pattern = 'registried_dwi_parenchyma.nii.gz', 
        mask_label=1
    ):
    for pid in tqdm(os.listdir(dwi_root)):
        try:
            dwi_file = os.path.join(dwi_root, pid, dwi_pattern)
            mask_file = os.path.join(mask_root, pid, mask_pattern)
            if not os.path.isfile(dwi_file):
                continue
            if not os.path.isfile(mask_file):
                continue
            dwi_image = DataIO.load_nii_image(dwi_file)['sitk_image']
            mask_image = DataIO.load_nii_image(mask_file)['sitk_image']
            extracted_dwi_image = ImagePostProcessingUtils.extract_region_by_mask(dwi_image, mask_image, default_value=-1024, mask_label=mask_label)
            
            # 将脑实质中小于0的值,设置为-1024，避免造成干扰
            tmp_arr = sitk.GetArrayFromImage(extracted_dwi_image)
            tmp_arr[tmp_arr<0] = -1024
            extracted_dwi_image = sitk.GetImageFromArray(tmp_arr)
            extracted_dwi_image.CopyInformation(dwi_image)

            out_sub_dir = os.path.join(out_root, pid)
            os.makedirs(out_sub_dir, exist_ok=True)
            out_dwi_file = os.path.join(out_sub_dir, out_dwi_pattern)

            sitk.WriteImage(extracted_dwi_image, out_dwi_file)
        except Exception as e:
            print(e)
            print('====> Error case:\t{}'.format(pid))

def merge_cerebral_parenchyma_mask_and_dwi_bbox(
        parenchyma_mask_root, 
        dwi_bbox_mask_root, 
        out_root, 
        parenchyma_mask_pattern='MASK.nii.gz',
        dwi_mask_pattern='registried_dwi_bbox.nii.gz',
        out_mask_pattern='final_mask.nii.gz'
    ):    
    for pid in tqdm(os.listdir(dwi_bbox_mask_root)):
        try:
            parenchyma_mask_file = os.path.join(parenchyma_mask_root, pid, parenchyma_mask_pattern)
            dwi_bbox_mask_file = os.path.join(dwi_bbox_mask_root, pid, dwi_mask_pattern)
            parenchyma_mask_image = sitk.ReadImage(parenchyma_mask_file)
            dwi_bbox_mask_image = sitk.ReadImage(dwi_bbox_mask_file)
            parenchyma_mask_arr = sitk.GetArrayFromImage(parenchyma_mask_image)
            dwi_bbox_mask_arr = sitk.GetArrayFromImage(dwi_bbox_mask_image)
            merged_mask_arr = parenchyma_mask_arr * dwi_bbox_mask_arr
            merged_mask_arr = np.array(merged_mask_arr, np.uint8)
            merged_mask_image = sitk.GetImageFromArray(merged_mask_arr)
            merged_mask_image.CopyInformation(parenchyma_mask_image)
            
            out_sub_dir = os.path.join(out_root, pid)
            os.makedirs(out_sub_dir, exist_ok=True)
            out_mask_file = os.path.join(out_sub_dir, out_mask_pattern)

            sitk.WriteImage(merged_mask_image, out_mask_file)            
        except Exception as e:
            print(e)
            print('====> Error case:\t{}'.format(pid))

def copy_train_data(data_root, out_root, cta_pattern='fixed_cta.nii.gz'):
    '''
    很多数据的分辨率太低，将能达到要求的数据copy到另外的文件夹
    '''
    os.makedirs(out_root, exist_ok=True)
    min_z = 10000
    max_z = 0
    for pid in tqdm(os.listdir(data_root)):
        cta_file = os.path.join(data_root, pid, cta_pattern)
        cta_image = sitk.ReadImage(cta_file)
        size = cta_image.GetSize()
        print('{}\t{}'.format(pid, size))
        if size[2] < 100:
            continue
        if min_z > size[2]:
            min_z = size[2]
        if max_z < size[2]:
            max_z = size[2]
        src_file = os.path.join(data_root, pid)
        dst_file = os.path.join(out_root, pid)
        shutil.copytree(src_file, dst_file)
    print('min z:\t{},\t\tmax z:\t{}'.format(min_z, max_z))

def data_preprocessing():
    data_root = '/data/medical/brain/gan/cta2dwi_multi_classified'

    # step_2_dcm_to_nii(os.path.join(data_root, '0.ori'), 
    #     os.path.join(data_root, '3.sorted_nii'))

    # step 3 cerebral parenchyma segmentation
    # cerebral_parenchyma_segmentation_new_algo(
    #     os.path.join(data_root, '3.sorted_nii'), 
    #     os.path.join(data_root, '3.sorted_mask')
    # )
    # step_3_3_segment_cerebral_parenchyma_connected_region(
    #     os.path.join(data_root, '3.sorted_mask')
    # )

    # extract_cta_cerebral_parenchyma_zlayers(
    #     os.path.join(data_root, '3.sorted_mask'), 
    #     os.path.join(data_root, '3.sorted_mask'), 
    #     os.path.join(data_root, '4.cropped_nii')
    # )

    # generate_dwi_bbox_mask(
    #     os.path.join(data_root, '3.sorted_nii'),
    #     os.path.join(data_root, '4.cropped_nii')
    # )

    # registration : run data_preprocessing_registration_dwi2cta.py

    # extract_dwi_cerebral_parenchyma(
    #     os.path.join(data_root, '4.registration_batch'), 
    #     os.path.join(data_root, '4.cropped_nii'), 
    #     os.path.join(data_root, '4.registration_batch')
    # )

    # merge_cerebral_parenchyma_mask_and_dwi_bbox(
    #     os.path.join(data_root, '4.cropped_nii'), 
    #     os.path.join(data_root, '4.registration_batch'), 
    #     os.path.join(data_root, '4.registration_batch')
    # )

    copy_train_data(
            os.path.join(data_root, '4.registration_batch'), 
            os.path.join(data_root, '5.train_batch')
        )




if __name__ == '__main__':
    # step_1_check_folder_format('/data/medical/brain/gan/hospital_6_multi_classified/CTA2DWI-多中心-20201102', 
    #     '/data/medical/brain/gan/cta2dwi_multi_classified')

    data_preprocessing()