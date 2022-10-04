import random, os

# order = list(range(14))
# random.seed(928)
# classes = os.listdir('/data1/fashion/FashionStyle14Refined/dataset')
# split = [0.7, 0.8]
# for i in range(10):
#     random.shuffle(classes)
#     print(classes)
#     tr = open('/data1/fashion/FashionStyle14Refined/imbalanced_splits/split_%d_train.csv' % i, 'w')
#     va = open('/data1/fashion/FashionStyle14Refined/imbalanced_splits/split_%d_val.csv' % i, 'w')
#     te = open('/data1/fashion/FashionStyle14Refined/imbalanced_splits/split_%d_test.csv' % i, 'w')
#     for o, cl in enumerate(classes):
#         if o == 0:
#             num_instances = len(os.listdir('/data1/fashion/FashionStyle14Refined/dataset/%s' % cl))
#         cnt = num_instances // (o + 1)
#         files = os.listdir('/data1/fashion/FashionStyle14Refined/dataset/%s' % cl)
#         random.shuffle(files)
#         for f in files[:int(split[0] * cnt)]:
#             tr.write('dataset/%s/%s\n' % (cl, f))
#         for f in files[int(split[0] * cnt):int(split[1] * cnt)]:
#             va.write('dataset/%s/%s\n' % (cl, f))
#         for f in files[int(split[1] * cnt):int(cnt)]:
#             te.write('dataset/%s/%s\n' % (cl, f))
#         print('SPLIT %d ORDER %d CLASS %s TR %d VA %d TE %d' %
#               (i, o, cl, int(split[0] * cnt), int((split[1] - split[0]) * cnt), int((1 - split[1]) * cnt)))

random.seed(928)
split = [0.7, 0.8]
classes = ['natural', 'mode', 'street', 'kireime-casual', 'rock', 'ethnic', 'retro', 'feminine', 'conservative',
		   'fairy', 'gal', 'lolita', 'dressy', 'girlish']
for i in range(100, 115):
    if i < 105:
        ratio = [1.0, 0.79, 0.61, 0.42, 0.37, 0.23, 0.20, 0.13, 0.09, 0.08, 0.08, 0.05, 0.02, 0.02]
    elif i < 110:
        ratio = [1 / (o + 1) for o in range(14)]
    else:
        ratio = [float(len(classes) - o) / len(classes) for o in range(14)]
    tr = open('/data1/fashion/FashionStyle14Refined/imbalanced_splits/split_%d_train.csv' % i, 'w')
    va = open('/data1/fashion/FashionStyle14Refined/imbalanced_splits/split_%d_val.csv' % i, 'w')
    te = open('/data1/fashion/FashionStyle14Refined/imbalanced_splits/split_%d_test.csv' % i, 'w')
    for o, cl in enumerate(classes):
        if o == 0:
            num_instances = len(os.listdir('/data1/fashion/FashionStyle14Refined/dataset/%s' % cl))
        cnt = num_instances * ratio[o]
        files = os.listdir('/data1/fashion/FashionStyle14Refined/dataset/%s' % cl)
        random.shuffle(files)
        for f in files[:int(split[0] * cnt)]:
            tr.write('dataset/%s/%s\n' % (cl, f))
        for f in files[int(split[0] * cnt):int(split[1] * cnt)]:
            va.write('dataset/%s/%s\n' % (cl, f))
        for f in files[int(split[1] * cnt):int(cnt)]:
            te.write('dataset/%s/%s\n' % (cl, f))
        print('SPLIT %d ORDER %d CLASS %s TR %d VA %d TE %d' %
              (i, o, cl, int(split[0] * cnt), int((split[1] - split[0]) * cnt), int((1 - split[1]) * cnt)))
