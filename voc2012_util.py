import os
import shutil

label_path = "D:/Study/VOC2012/ImageSets/Main"
output_label = "D:/Study/VOC2012/labels.txt"
output_path = "D:/Study/VOC2012/Classification"
image_folder = "D:/Study/VOC2012/JPEGImages"
dataset_split = ['train', 'val', 'test']

def get_class(path):
    labels = set()
    file_names = os.listdir(path)
    for file_name in file_names:
        tmp = file_name.split('.')
        tmp = tmp[0].split('_')
        if (len(tmp) < 2):
            continue
        labels.add(tmp[0])
    return labels

def set_label(label_dict, path, label):
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        tmp = line.split('\n')
        tmp = tmp[0].strip().split(' ')
        file_name = tmp[0]
        if int(tmp[-1]) == 1:
            label_dict[file_name] = label

def generate_label(input_path, output_txt_path, write_file = True):
    label_dict = {}
    file_names = os.listdir(input_path)
    for file_name in file_names:
        if "trainval" in file_name:
            tmp = file_name.split('.')
            tmp = tmp[0].split('_')
            if (len(tmp) < 2):
                continue
            set_label(label_dict, os.path.join(input_path, file_name), tmp[0])

    if write_file:
        f = open(output_txt_path, 'w')
        lines = []
        for key, val in label_dict.items():
            lines.append("{},{}\n".format(key, val))
        
        f.writelines(lines)
        f.close()

    return label_dict

def split_image(label_dict, output_folder, image_folder):
    file_names = os.listdir(image_folder)
    count = 0
    for image_file in file_names:
        base_name, _ = os.path.splitext(image_file)
        if base_name in label_dict:
            src = os.path.join(image_folder, image_file)
            if count % 10 < 6:
                dst = os.path.join(output_folder, 'train', label_dict[base_name], image_file)
                shutil.copy(src, dst)
            elif 6 < count % 10 < 9:
                dst = os.path.join(output_folder, 'val', label_dict[base_name], image_file)
                shutil.copy(src, dst)
            else:
                dst = os.path.join(output_folder, 'test', label_dict[base_name], image_file)
                shutil.copy(src, dst)
            count+=1
    
if __name__ == '__main__':
    
    labels = get_class(label_path)
    print(labels)
    # for sp in dataset_split:
    #     cur_dir = os.path.join(output_path, sp)
    #     for label in labels:
    #         label_dir = os.path.join(cur_dir, label)
    #         os.mkdir(label_dir)
    label_dict = generate_label(label_path, output_path, False)
    split_image(label_dict, output_path, image_folder)

        