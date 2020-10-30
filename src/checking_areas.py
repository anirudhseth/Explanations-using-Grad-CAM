import os
import pascal_voc_tools
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

folder = 'ILSVRC2012_bbox_val_v3/ILSVRC2012_bbox_val_v3/val'

n = 10000
annotations = next(os.walk(folder))[2][:n]
parser = pascal_voc_tools.XmlParser()
frac_list = []

for annotation in tqdm(annotations):
    obj_list = parser.load(os.path.join(folder,annotation))

    size = obj_list['size']

    for obj in obj_list['object']:
        
        bb = obj['bndbox']

        area_img = int(size['width'])*int(size['height'])
        area_bb = (int(bb['xmax']) - int(bb['xmin']))*(int(bb['ymax']) - int(bb['ymin']))

        frac = area_bb/area_img

        frac_list.append(frac)

tot = np.sum(np.array(frac_list) > 0.5)
plt.hist(frac_list,50)
plt.show()
print(np.mean(frac_list))
print(tot/n)