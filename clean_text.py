import re

def clean_text(text):
  text = text.lower()
  text = re.sub('[^A-Za-z0-9]+', ' ', text)
  return text