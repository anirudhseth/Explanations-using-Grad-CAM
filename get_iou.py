import numpy as np
import os
import matplotlib.pyplot as plt

folder = r'C:\Users\magnu\OneDrive\Dokument\KTH\2020HT\DD2412_-_Deep_Learning_Advanced_Course\Project\Results\resnet'

files = next(os.walk(folder))[2]

for filename in files:
    if os.path.splitext(filename)[0][-3:] == 'iou':
        filepath = os.path.join(folder,filename)
        print(filename)

        f = open(filepath,'r')

        null_ious = 0
        ious = []

        lines = f.readlines()

        for line in lines:
            iou = float(line)

            if iou > 0:
                ious.append(iou)
            else:
                null_ious += 1

        
        print(np.mean(ious))
        print(np.std(ious))