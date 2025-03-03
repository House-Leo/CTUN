from os import path as osp
import glob
import shutil

lq_root = '/data1/lihao/datasets/VSR/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
gt_root = '/data1/lihao/datasets/VSR/vimeo90k/vimeo_septuplet/sequences'
meta_info = 'basicsr/data/meta_info/meta_info_Vimeo90K_test_GT.txt'
# num_frame = 7
# neighbor_list = [i + (9 - num_frame) // 2 for i in range(num_frame)]
lq_save_path = '/data1/lihao/datasets/VSR/vimeo90k/vimeo_test_lq'
gt_save_path = '/data1/lihao/datasets/VSR/vimeo90k/vimeo_test_gt'

with open(meta_info, 'r') as fin:
    subfolders = [line.split(' ')[0] for line in fin]

for idx, subfolder in enumerate(subfolders):
    gt_path = osp.join(gt_root, subfolder, 'im4.png')
    lq_path = osp.join(lq_root, subfolder, 'im4.png')
    split_result = lq_path.split('/')
    img_name = f'{split_result[-3]}_{split_result[-2]}'
    lq_save_img_path = osp.join(lq_save_path, f'{img_name}.png')
    gt_save_img_path = osp.join(gt_save_path, f'{img_name}.png')
    shutil.copy(lq_path,lq_save_img_path)
    shutil.copy(gt_path,gt_save_img_path)
    # print(lq_save_img_path)
    # print(lq_path)
    # print(gt_path)
    # if idx>3:
    #     break
