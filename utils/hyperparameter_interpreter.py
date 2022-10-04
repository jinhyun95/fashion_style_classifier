import os
import re
import numpy as np

basedir = '/data5/assets/jinhyun95/FashionStyle'
base_name = ['_MobileNet', '_ResNet18', '_ResNet34','_ResNeXt50', '_ResNet50', '_ResNet101', '_WideResNet50']
# base_name = ['ResNet50']
# split = 114
result = []
timestamp = 'Hyperparam'
for expname in sorted(os.listdir(basedir)):
    if timestamp in expname:
        try:
            exp = os.path.join(basedir, expname, 'logs')
            logfiles = os.listdir(exp)
            logfiles = sorted(logfiles, key = lambda x: int(re.search('run_sequence_([0-9]+)_log.txt', x).group(1)), reverse=True)
            order = ['top_1_any_accuracy', 'top_3_all_accuracy']
            performance = {k: 0 for k in order}
            class_statistics = []
            for last in logfiles:
                try:
                    start_test = False
                    with open(os.path.join(exp, last), 'r') as f:
                        for line in f.readlines():
                            if 'TEST RESULT' in line:
                                search = re.search('TEST RESULT ([a-zA-Z\_0-9]+) ([0-9\.]+)', line)
                                performance[search.group(1)] = '%.2f' % (float(search.group(2)) * 100.)
                                if search.group(1) == 'top_3_all_accuracy':
                                    start_test = True
                            search = re.search('\s+([a-zA-Z_]+)\s+([0-9]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)', line)
                            if start_test and search is not None:
                                class_statistics.append([float(x) for x in [search.group(2), search.group(3), search.group(4), search.group(5), search.group(6), search.group(7)]])
                    break
                except:
                    pass
            class_statistics.sort(key=lambda x: x[0], reverse=True)
            class_statistics = np.array(class_statistics)[:, 1:]
            result.append([expname])
            # for o in order:
            #     result[-1].append(performance[o])
            for i in range(1, 3):
                result[-1].append('%.2f' % np.mean(class_statistics[:, i]))
        except:
            pass
try:
    full_atts = max([len(x) for x in result])
    filtered_result = []
    for r in result:
        if len(r) == full_atts:
            filtered_result.append(r)
    for i in range(len(filtered_result[0])):
        print(' '.join([str(r[i]) for r in filtered_result]))
except:
    pass
