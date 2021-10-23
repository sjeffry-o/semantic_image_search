import telebot
import ngtpy
from token_tg import TOKEN
from text_clip_model import load_text_model
from clip.evaluate.utils import get_text_batch

text_model, tokenizer, args = load_text_model()
bot = telebot.TeleBot(TOKEN)
index = ngtpy.Index(b"./flickr_artworks_albumCovers_movieposters_dupfree")
args.cpu = True

@bot.message_handler(content_types=["text"])
def generate(message):
  new_text = message.text
  chatid = message.chat.id
  input_ids, attention_mask = get_text_batch([new_text], tokenizer, args)
  vector = text_model(**{"x": input_ids, "attention_mask": attention_mask}).to('cpu').detach().numpy()
  results = index.search(vector, 6)
  for i, (id, distance) in enumerate(results):
      print(str(i) + ": " + str(id) + ", " + str(distance))
      object = index.get_object(id)
      print(object)
      img = open(img_paths[id], 'rb')
      try:
        bot.send_photo(chatid, img)
      except:
        bot.send_photo(chatid, img)

if __name__ == '__main__':
    while True:
      try:
          bot.polling(none_stop=True)
      except Exception as e:
          print(e) 
          time.sleep(15)
