#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_preprocessing_registration.py
@Time    :   2021/07/14 15:27:14
@Author  :   searobbersandduck 
@Version :   1.0
@Contact :   searobbersandduck@gmail.com
@License :   (C)Copyright 2020-2021, MIT
@Desc    :   None
'''

'''
由于环境安装问题，目前该代码需切换到elastix安装路径下执行。
cp data_preprocessing_registration_dwi2cta.py /home/zhangwd/code/pkg/SimpleElastix/build/SimpleITK-build/Wrapping/Python/
cd /home/zhangwd/code/pkg/SimpleElastix/build/SimpleITK-build/Wrapping/Python/

source activate pytorch1.6
python data_preprocessing_registration.py
'''

import os
import sys
from glob import glob
from tqdm import tqdm
import time
import numpy as np

import SimpleITK as sitk

# 情况特殊，此处写绝对路径
sys.path.append('/home/zhangwd/code/work')
from MedCommon.utils.data_io_utils import DataIO

selx = sitk.ElastixImageFilter()
print('\n'.join(dir(selx)))


def elastix_register_images_one_case(cta_file, dwi_file, dwi_bbox_file, out_dir, is_dcm=False):
    
    cta_data = DataIO.load_nii_image(cta_file)
    dwi_data = DataIO.load_nii_image(dwi_file)
    dwi_bbox_data = DataIO.load_nii_image(dwi_bbox_file)

    cta_img = cta_data['sitk_image']
    dwi_img = dwi_data['sitk_image']
    dwi_bbox_img = dwi_bbox_data['sitk_image']

    # cta_img.SetOrigin([0,0,0])
    dwi_img.SetOrigin(cta_img.GetOrigin())
    dwi_bbox_img.SetOrigin(cta_img.GetOrigin())

    selx = sitk.ElastixImageFilter()
    print(cta_img.GetSize())
    print(dwi_img.GetSize())
    selx.SetFixedImage(cta_img)
    selx.SetMovingImage(dwi_img)
    selx.SetParameterMap(selx.GetDefaultParameterMap('rigid'))
    selx.Execute()

    moved_dwi_img = sitk.Transformix(dwi_img, selx.GetTransformParameterMap())
    moved_dwi_img.CopyInformation(cta_img)
    moved_dwi_bbox_img = sitk.Transformix(dwi_bbox_img, selx.GetTransformParameterMap())
    moved_dwi_bbox_img.CopyInformation(cta_img)

    os.makedirs(out_dir, exist_ok=True)
    out_cta_file = os.path.join(out_dir, 'fixed_cta.nii.gz')
    out_registried_dwi_file = os.path.join(out_dir, 'registried_dwi.nii.gz')
    out_registried_dwi_bbox_file = os.path.join(out_dir, 'registried_dwi_bbox.nii.gz')   


    print('{}:\t{}'.format(out_registried_dwi_file, moved_dwi_img.GetSize()))
    print('{}:\t{}'.format(out_registried_dwi_bbox_file, moved_dwi_bbox_img.GetSize()))

    sitk.WriteImage(cta_img, out_cta_file)
    sitk.WriteImage(moved_dwi_img, out_registried_dwi_file)
    sitk.WriteImage(moved_dwi_bbox_img, out_registried_dwi_bbox_file)

def test_elastix_register_images_one_case():
    pid = '4730391'
    data_root = '/data/medical/brain/gan/cta2dwi_multi_classified/4.cropped_nii'
    
    pid_path = os.path.join(data_root, pid)
    cta_file = os.path.join(pid_path, 'CTA.nii.gz')
    dwi_file = os.path.join(pid_path, 'DWI.nii.gz')
    dwi_bbox_file = os.path.join(pid_path, 'DWI_BBOX_MASK.nii.gz')

    out_dir = '/data/medical/brain/gan/cta2dwi_multi_classified/4.registration_test/{}'.format(pid)

    beg = time.time()
    # register_images(cta_file, mip_file, bf_file, True)
    elastix_register_images_one_case(cta_file, dwi_file, dwi_bbox_file, out_dir, False)
    end = time.time()
    print('====> test_register_images time cosume is:\t{:.3f}'.format(end-beg))

def elastix_register_images_single_task(data_root, out_dir, pids, task_id):
    log = []
    for pid in tqdm(pids):
        try:
            pid_path = os.path.join(data_root, pid)
            cta_file = os.path.join(pid_path, 'CTA.nii.gz')
            dwi_file = os.path.join(pid_path, 'DWI.nii.gz')
            dwi_bbox_file = os.path.join(pid_path, 'DWI_BBOX_MASK.nii.gz')
            out_pid_dir = os.path.join(out_dir, pid)

            elastix_register_images_one_case(cta_file, dwi_file, dwi_bbox_file, out_pid_dir, False)
            
            
        except Exception as e:
            print(e)
            print('====> Error case:\t{}'.format(pid))
            log.append(e)
            log.append('====> Error case:\t{}'.format(pid))

    with open('log_{}'.format(task_id), 'w') as f:
        f.write('\n'.join(log))

def elastix_register_images_multi_task(data_root, out_dir, process_num=6, reuse=False):
    pids = []
    gen_pids = []
    for pid in os.listdir(out_dir):
        gen_pids.append(pid)

    for pid in os.listdir(data_root):
        if len(pid) == 7:
            if reuse:
                if pid not in gen_pids:
                    pids.append(pid)
            else:
                pids.append(pid)
    
    num_per_process = (len(pids) + process_num - 1)//process_num

    # this for single thread to debug
    # elastix_register_images_single_task(data_root, out_dir, pids)

    # this for run 
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()

    results = []

    print(len(pids))
    for i in range(process_num):
        sub_pids = pids[num_per_process*i:min(num_per_process*(i+1), len(pids))]
        print(len(sub_pids))
        result = pool.apply_async(elastix_register_images_single_task, 
            args=(data_root, out_dir, sub_pids, i))
        results.append(result)

    pool.close()
    pool.join()

def test_elastix_register_images_multi_task():
    data_root = '/data/medical/brain/gan/cta2dwi_multi_classified/4.cropped_nii'
    out_dir = '/data/medical/brain/gan/cta2dwi_multi_classified/4.registration_batch'   
    os.makedirs(out_dir, exist_ok=True)    
    elastix_register_images_multi_task(data_root, out_dir, 6, reuse=True)

if __name__ == '__main__':
    # test_elastix_register_images_one_case()
    test_elastix_register_images_multi_task()

