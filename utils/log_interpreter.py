import os, re

basedir = '/data3/assets/jinhyun95/FashionStyle'
regex = 'KFashion.+(anchor|bce)'
experiments = os.listdir(basedir)
for exp in experiments:
    if re.match(regex, exp) is not None and not exp.endswith('.csv'):
        log = os.path.join(basedir, exp, 'logs')
        summary = os.path.join(basedir, log.split('/')[-2] + '_summary.csv')
        out = open(summary, 'w')
        logfiles = os.listdir(log)
        logfiles = sorted(logfiles, key = lambda x: int(re.search('run_sequence_([0-9]+)_log.txt', x).group(1)), reverse=True)
        for last in logfiles:
            start_test = False
            with open(os.path.join(log, last), 'r') as f:
                for line in f.readlines():
                    if 'TEST RESULT' in line:
                        start_test = True
                        search = re.search('TEST RESULT ([a-zA-Z\_0-9]+) ([0-9\.]+)', line)
                        out.write('\n\n\n%s, %.2f\n' % (search.group(1), float(search.group(2)) * 100.))
                        print('%s, %.2f' % (search.group(1), float(search.group(2)) * 100.))
                        out.write('Class Statistics\n')
                        print('Class Statistics')
                        out.write('Style, image count, top 1 recall, top 2 recall, top 3 recall, top 4 recall, top 5 recall\n')
                        print('Style, image count, top 1 recall, top 2 recall, top 3 recall, top 4 recall, top 5 recall')
                    if start_test:
                        search = re.search('\s+([a-zA-Z_]+)\s+([0-9]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)', line)
                        if search is not None:
                            out.write('%s, %s, %s, %s, %s, %s, %s\n' %
                                    (search.group(1), search.group(2), search.group(3), search.group(4), search.group(5), search.group(6), search.group(7)))
                            print('%s, %s, %s, %s, %s, %s, %s' %
                                  (search.group(1), search.group(2), search.group(3), search.group(4), search.group(5), search.group(6), search.group(7)))
                f.close()
            if start_test:
                out.close()
                break