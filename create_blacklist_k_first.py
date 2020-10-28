# Create blacklist for 100 and 1000 first values in ILSVRC2015 clsloc competition

def create_blacklist_k_first(blacklist,ground_truth,k):

    bl_file = open(blacklist,'r')
    ground_truth = open(ground_truth,'r')
    new_file = open(f'ILSVRC2015_devkit.tar\ILSVRC2015_devkit\ILSVRC2015\devkit\data\ILSVRC2015_clsloc_validation_blacklist_first_{k}.txt','w')

    bl_lines = bl_file.readlines()
    gt_lines = ground_truth.readlines()

    bl_list = []
    gt_list = []

    k_first_bl_list = []

    for bl_line in bl_lines:
        bl_idx_int = int(bl_line.strip())
        if bl_idx_int > k:
            break
        else:
            new_file.write(bl_line)

    new_file.close()

    print(len(bl_list))

bl = 'ILSVRC2015_devkit.tar\ILSVRC2015_devkit\ILSVRC2015\devkit\data\ILSVRC2015_clsloc_validation_blacklist.txt'
gt = 'ILSVRC2015_devkit.tar\ILSVRC2015_devkit\ILSVRC2015\devkit\data\ILSVRC2015_clsloc_validation_ground_truth.txt'

create_blacklist_k_first(bl,gt,100)