# Create dict with bbox info of k first imgs in ilsvrc2012 data

import os
import pascal_voc_tools
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

folder = 'ILSVRC2012_bbox_val_v3/ILSVRC2012_bbox_val_v3/val'

n = 1000

annotations = next(os.walk(folder))[2][:n]
parser = pascal_voc_tools.XmlParser()
bb_dict = {}

for annotation in tqdm(annotations):
    obj_list = parser.load(os.path.join(folder,annotation))
    
    size = obj_list['size']

    w = int(size['width'])
    h = int(size['height'])

    name = obj_list['filename']
    bb_dict[name] = {'size':(w,h),'bndboxes':[]}
    bb_list = []
    for obj in obj_list['object']:
        

        bb = obj['bndbox']
        xmin = int(bb['xmin'])
        ymin = int(bb['ymin'])
        xmax = int(bb['xmax'])
        ymax = int(bb['ymax'])
        tl = (xmin,ymin)
        br = (xmax,ymax)
        bb_dict[name]['bndboxes'].append((tl,br))

pickle.dump(bb_dict,open(f'bb_dict_{n}.p','wb'))