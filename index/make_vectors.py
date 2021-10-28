from glob import glob
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import os
import sys
sys.path.append("/root/semantic_image_search/ru-clip")
from clip.evaluate.utils import load_weights_only
import pickle
import torch

def load_visual_model():
    model, _ = load_weights_only("ViT-B/32-small")
    visual_model = model.visual_encoder.float().eval()

    del model
    return visual_model

def make_vectors(imgs_root_path = '/root/semantic_image_search/book_covers'):
    img_paths = sorted(glob(os.path.join(imgs_root_path, '*jpg')))
    img_transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    img_vectors = torch.Tensor().to('cpu')
    count = 0
    device = 'cuda'
    vis_model = load_visual_model()
    with torch.no_grad():
      for path in img_paths:
        if count % 10000 == 0:
          print(count, "passed")
        image = Image.open(path)
        image = image.convert("RGB")
        image = img_transform(image)
        image = torch.tensor(image)
        image = image.to(device)
        img_vectors = torch.cat([img_vectors, vis_model(image.unsqueeze(0)).to('cpu')])
        count += 1
    return img_vectors

if __name__ == '__main__':
    vectors = make_vectors()
    with open('img_vectors.pickle', 'wb') as f:
        pickle.dump(vectors, f)
