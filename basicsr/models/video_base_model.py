import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class VideoBaseModel(SRModel):
    """Base video SR model."""

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        # num_folders = len(dataset)
        # num_pad = (world_size - (num_folders % world_size)) % world_size
        # if rank == 0:
        #     pbar = tqdm(total=len(dataset), unit='folder')
        # # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # # (To avoid wait-dead)
        # for i in range(rank, num_folders + num_pad, world_size):
        #     idx = min(i, num_folders - 1)
        #     val_data = dataset[idx]
        #     folder = val_data['folder']

        #     # compute outputs
        #     val_data['lq'].unsqueeze_(0)
        #     val_data['gt'].unsqueeze_(0)
        #     self.feed_data(val_data)
        #     val_data['lq'].squeeze_(0)
        #     val_data['gt'].squeeze_(0)

        #     self.test()
        #     visuals = self.get_current_visuals()

        #     # tentative for out of GPU memory
        #     del self.lq
        #     del self.output
        #     if 'gt' in visuals:
        #         del self.gt
        #     torch.cuda.empty_cache()

        #     if self.center_frame_only:
        #         visuals['result'] = visuals['result'].unsqueeze(1)
        #         if 'gt' in visuals:
        #             visuals['gt'] = visuals['gt'].unsqueeze(1)

        #     # evaluate
        #     if i < num_folders:
        #         for idx in range(visuals['result'].size(1)):
        #             result = visuals['result'][0, idx, :, :, :]
        #             result_img = tensor2img([result])  # uint8, bgr
        #             metric_data['img'] = result_img
        #             if 'gt' in visuals:
        #                 gt = visuals['gt'][0, idx, :, :, :]
        #                 gt_img = tensor2img([gt])  # uint8, bgr
        #                 metric_data['img2'] = gt_img

        #             if save_img:
        #                 if self.opt['is_train']:
        #                     raise NotImplementedError('saving image is not supported during training.')
        #                 else:
        #                     if self.center_frame_only:  # vimeo-90k
        #                         clip_ = val_data['lq_path'].split('/')[-3]
        #                         seq_ = val_data['lq_path'].split('/')[-2]
        #                         name_ = f'{clip_}_{seq_}'
        #                         img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
        #                                             f"{name_}_{self.opt['name']}.png")
        #                     else:  # others
        #                         img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
        #                                             f"{idx:08d}_{self.opt['name']}.png")
        #                     # image name only for REDS dataset
        #                 imwrite(result_img, img_path)

        #             # calculate metrics
        #             if with_metrics:
        #                 for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
        #                     result = calculate_metric(metric_data, opt_)
        #                     self.metric_results[folder][idx, metric_idx] += result

        #         # progress bar
        #         if rank == 0:
        #             for _ in range(world_size):
        #                 pbar.update(1)
        #                 pbar.set_description(f'Folder: {folder}')

        # if rank == 0:
        #     pbar.close()

        # if with_metrics:
        #     if self.opt['dist']:
        #         # collect data among GPUs
        #         for _, tensor in self.metric_results.items():
        #             dist.reduce(tensor, 0)
        #         dist.barrier()

        #     if rank == 0:
        #         self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        # # record all frames (border and center frames)
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='frame')
        for idx in range(rank, len(dataset), world_size):
            val_data = dataset[idx]
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            folder = val_data['folder']
            frame_idx, max_idx = val_data['idx'].split('/')
            lq_path = val_data['lq_path']

            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            result_img = tensor2img([visuals['result']])
            metric_data['img'] = result_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    raise NotImplementedError('saving image is not supported during training.')
                else:
                    if 'vimeo' in dataset_name.lower():  # vimeo90k dataset
                        split_result = lq_path.split('/')
                        img_name = f'{split_result[-3]}_{split_result[-2]}_{split_result[-1].split(".")[0]}'
                    else:  # other datasets, e.g., REDS, Vid4
                        img_name = osp.splitext(osp.basename(lq_path))[0]

                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                 f'{img_name}.png')
                imwrite(result_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                    result = calculate_metric(metric_data, opt_)
                    self.metric_results[folder][int(frame_idx), metric_idx] += result

            # progress bar
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {folder}: {int(frame_idx) + world_size}/{max_idx}')
        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()
            else:
                pass  # assume use one gpu in non-dist testing

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        # ----------------- calculate the average values for each folder, and for each metric  ----------------- #
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }
        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        total_avg_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)
            # update the best metric result
            self._update_best_metric_result(dataset_name, metric, total_avg_results[metric], current_iter)

        # ------------------------------------------ log the metric ------------------------------------------ #
        log_str = f'Validation {dataset_name}\n'
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\n\t    Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f'metrics/{metric}/{folder}', tensor[metric_idx].item(), current_iter)
