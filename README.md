# Explanations-using-GradCAM
Reimplimentation of the paper - Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

conda create --name <env> --file requirements.txt

# RESULTS
## ILSVRC2015 Localization Challenge

## Grad-CAM and Grad-CAM++ visualisations
<img src="Results/GC_GCP_compare.jpeg" width="80%" height="80%" ></img>

### Generated Bounding Boxes
<img src="Results/imgs_[297 254 574 913 443]_bbs_vgg16.png" width="80%" height="80%" ></img>
<img src="Results/imgs_[309 405 388 789 721]_bbs_vgg16.png" width="80%" height="80%" ></img>
<img src="Results/imgs_[437 851 810 638 802]_bbs_vgg16.png" width="80%" height="80%" ></img>

### Bounding Boxes Generation process
<img src="Results/combined_expl_bb_ILSVRC2012_val_00000443.png" width="80%" height="80%" ></img>
<img src="Results/combined_expl_bb_ILSVRC2012_val_00000721.png" width="80%" height="80%" ></img>
<img src="Results/combined_expl_bb_ILSVRC2012_val_00000802.png" width="80%" height="80%" ></img>

### Results

<img src="Results/table1.PNG" width="80%" height="80%" ></img>
<img src="Results/table2.PNG" width="80%" height="80%" ></img>

## Counterfactual Explanations using Grad-CAM
<img src="Results/CounterFactExp-crop.jpeg"></img>
## Image Captioning visualisations using Grad-CAM
<img src="Results/Caption1.PNG" width="50%" height="50%"></img> 
<img src="Results/Caption2.PNG" width="50%" height="50%"></img>
<img src="Results/Caption3.PNG" width="50%" height="50%"></img>
## Effect of adveserial noise on Grad-CAM
<img src="Results/Effect_of_Noise1.jpeg"></img>
## Robust Grad-CAM
<img src="Results/Robust_architcture_results-crop.jpeg"></img>
## Mode Collapse detection using Grad-CAM
<img src="Results/Input%20image%20for%20mode%20collapse%20example.png" width="30%" height="30%"></img>
<img src="Results/dcgan_heatmap_3.gif" width="30%" height="30%"></img>
