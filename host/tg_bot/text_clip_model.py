import sys
sys.path.append("/root/semantic_image_search/ru-clip")
from clip.evaluate.utils import (get_tokenizer, load_weights_only)

def load_text_model():
    model, args = load_weights_only("ViT-B/32-small")
    text_model = model.text_encoder.float().eval()
    tokenizer = get_tokenizer()

    del model
    return text_model, tokenizer, args
