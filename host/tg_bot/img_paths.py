from glob import glob
import os

def del_flickr_duplicates(img_paths):
    count = 0
    del_idxs = []
    for path in img_paths:
      if path.count("flickr30k_images") == 3:
        del_idxs.append(count)
      count += 1
    start, end = del_idxs[0], del_idxs[-1]
    img_paths = img_paths[:start] + img_paths[end + 1:]
    return img_paths

def img_paths(root_path):
    img_paths_1 = [img_path for img_path in glob(os.path.join(root_path, '*/*jpg'))]
    img_paths_2 = [img_path for img_path in glob(os.path.join(root_path, '*/*/*jpg'))]
    img_paths_3 = [img_path for img_path in glob(os.path.join(root_path, '*/*/*/*jpg'))]
    img_paths = sorted(img_paths_1 + img_paths_2 + img_paths_3)
    img_paths = del_flickr_duplicates(img_paths)
    return img_paths
