import os
import torch
from collections import OrderedDict
import glob
import logging
from tqdm import tqdm
_join = os.path.join


class Saver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.run_path, args.dataset, args.model)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        self.runs.sort(key=lambda x: int(x.split('_')[-1]))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        self.run_id = run_id

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            print('Create experiment_{}'.format(str(run_id)))

    def save_experiment_config(self, total_params):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        log_file.writelines('------------------ Important Parameters ------------------' + '\n')
        p = OrderedDict()
        p['func_main'] = self.args.func_main
        p['datset'] = self.args.dataset
        p['model'] = self.args.model
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        # p['grid_w'] = self.args.grid_w
        # p['grid_h'] = self.args.grid_h
        p['Total_params'] = total_params

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')

        log_file.writelines('\n' + '------------------ All Parameters ------------------' + '\n')
        for eachArg, value in self.args.__dict__.items():
            log_file.writelines(eachArg + ' : ' + str(value) + '\n')
        log_file.writelines('------------------- end -------------------')
        log_file.close()


def make_log(log_dir):
    logging.basicConfig(level=logging.INFO,
                        filename=_join(log_dir, 'new.log'),
                        filemode='w',
                        format='%(asctime)s - : %(message)s')

    logging.info('PyTorch Version:' + str(torch.__version__) + '\n')

    return logging


def myprint(logging, message, print_in_tqdm=False):
    logging.info(message)
    if print_in_tqdm:
        tqdm.write(message)  # tqdm conflicts with print
    else:
        print(message)
