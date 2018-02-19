import numpy as np
from elecparts_dataset import elecpartsDataset
from unreal_utils import compute_mean_AP_from_annotations

# The groundtruth
annot_gt_file = '/nas/datashare/datasets/elecparts/3/12objects_real/annotation.pkl'

# This results in a mAP of 0.79 
annot_detect_file = '/nas/datashare/datasets/elecparts/3/results/epoch_009/correct_rotation_and_our_resize/result_real.pkl'



dataset = elecpartsDataset()
class_2_num = dict(zip(dataset.my_class_names, np.arange(len(dataset.my_class_names))))


mAP, APs =compute_mean_AP_from_annotations(class_2_num, annot_detect_file, annot_gt_file)
print('mAP:', mAP)
