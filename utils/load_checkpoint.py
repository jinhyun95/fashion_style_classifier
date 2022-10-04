import os
import torch
import re
import sys

def load_checkpoint(args, net, optimizer, metric):
    base_dir = os.path.join(args.out_dir, args.exp_name)
    epoch = 0
    best = 0
    if args.load_checkpoint and os.path.exists(os.path.join(base_dir, 'checkpoints', 'last.pth')):
        checkpoint = torch.load(os.path.join(base_dir, 'checkpoints', 'last.pth'))
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        log_dir = os.path.join(base_dir, 'logs')
        for log in os.listdir(log_dir):
            if log == 'run_sequence_%d_log.txt' % len(os.listdir(log_dir)):
                continue
            with open(os.path.join(log_dir, log), 'r') as f:
                for line in f.readlines():
                    if 'VALIDATION LOSS:' in line:
                        epoch = max(epoch, int(re.search('EPOCH ([0-9]+) VALIDATION', line).group(1)))
                        best = max(best, float(re.search('%s: ([0-9\.]+),' % metric, line).group(1)))
        return epoch, best
    elif args.load_checkpoint:
        print('FAILED TO LOAD CHECKPOINT : CHECKPOINT DOES NOT EXIST')
        sys.exit()
    elif os.path.exists(os.path.join(base_dir, 'checkpoints', 'last.pth')):
        print('FAILED TO START TRAINING : EXPERIMENT DIRECTORY ALREADY EXISTS')
        sys.exit()
    else:
        return 0, 0