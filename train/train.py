import os
import shutil
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(ROOT)

sys.path.append(ROOT)

from external_lib.MedCommon.gan.runner.train_gan_3d import train, inference, GANTrainer

# remove inference error data
# 只保留mae<80 and  10<mask_mae<40

def remove_error_data(mae_csv_file, data_root):
    import pandas as pd
    df = pd.read_csv(mae_csv_file)
    df = df[~((df['mae']<200)&(df['mask_mae']>10) & (df['mask_mae']<40) & (df['mae']/df['mask_mae']<5))]
    error_list = df['suid'].tolist()
    for suid in error_list:
        try:
            error_case = os.path.join(data_root, str(suid))
            print('remove case:\t', suid)
            shutil.rmtree(error_case)
        except:
            pass


if __name__ == '__main__':
    train()
    # inference(
    #         '/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/5.train_batch_2d_parenchyma', 
    #         '/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/6.inference_352x352x192_eval', 
    #         '/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/checkpoints/cta2dwi_sr/480_net_G.pth'
    #     )
    # GANTrainer.calc_mae_with_mask(
    #         data_root='/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/6.inference_352x352x192_eval', 
    #         out_dir = '/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/7.analysis_result', 
    #         out_file = 'mae_352x352x192_eval.csv', 
    #         mask_root = '/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/5.train_batch_2d_parenchyma', 
    #         mask_pattern = 'final_mask.nii.gz', 
    #         mask_label = 1, 
    #         crop_size=[352,352,192]
    # )
    # remove_error_data('/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/7.analysis_result/mae_352x352x192_eval.csv', '/ssd/zhangwd/cta2mbf/cta2dwi_all_2d_parenchyma/5.train_batch_2d_parenchyma')


