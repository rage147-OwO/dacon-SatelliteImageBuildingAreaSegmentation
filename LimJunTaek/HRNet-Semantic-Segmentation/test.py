import os

ROOT = r'C:\Users\lim\Downloads\Cityscapes'

def convert_to_lst(root_dir, split):
    image_dir = os.path.join(root_dir, 'leftImg8bit', split)
    annotation_dir = os.path.join(root_dir, 'gtFine', split)
    lst_file = f'D:\GitHub\HRNet-Semantic-Segmentation\{split}.lst'

    with open(lst_file, 'w') as f:
        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_annotation_dir = os.path.join(annotation_dir, city)

            for filename in os.listdir(city_image_dir):
                image_path = os.path.join(city_image_dir, filename)
                annotation_filename = filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                annotation_path = os.path.join(city_annotation_dir, annotation_filename)

                line = f'{image_path}\t{annotation_path}\n'
                f.write(line)

    print(f'Created {lst_file} file.')

# Convert train set to lst
convert_to_lst(ROOT, 'train')

# Convert validation set to lst
convert_to_lst(ROOT, 'val')

# Convert test set to lst
#convert_to_lst(ROOT, 'test')
