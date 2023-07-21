from abc import ABC, abstractmethod
import re
import nltk
from nltk.corpus import stopwords , wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextProcessor(ABC):
  @abstractmethod
  def transform(self, text):
    pass

class ConvertCase(TextProcessor):
  def __init__(self, casing='lower'):
    self.casing = casing

  def transform(self, text):
    if self.casing == 'lower':
      return text.lower()
    if self.casing == 'upper':
      return text.upper()

class RemoveDigit(TextProcessor):
  def transform(self, text):
    return re.sub("[^a-zA-Z#]", " ", text)

class RemoveSpace(TextProcessor):
  def transform(self,text):
    return re.sub("\s+", " ", text).strip()

class RemoveUserHandle(TextProcessor):
  def transform(self,text):
    return re.sub("@[\w]*", "", text)

class Removehttplinks(TextProcessor):
  def transform(self,text):
    return re.sub("http\S+", "", text)

class Lemmatizer(TextProcessor):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_token(self, token):
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        pos = nltk.pos_tag([token])[0][1][0].upper()
        if pos in tag_dict.keys():
          wordnet_tag = tag_dict.get(pos)
          return self.lemmatizer.lemmatize(token, pos=wordnet_tag)
        else:
            return token

    def transform(self, text):
        tokens = nltk.word_tokenize(text)
        lemmatized_tokens = [self.lemmatize_token(token) for token in tokens]
        lemmatized_text = ' '.join(lemmatized_tokens)
        return lemmatized_text


class Stemmer(TextProcessor):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def transform(self, text):
        tokens = nltk.word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        stemmed_text = ' '.join(stemmed_tokens)
        return stemmed_text

class RemoveStopWords(TextProcessor):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def transform(self, text):
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text



class TextPreProcessPipeLine:
  def __init__(self, *args):
    self.transformers = args

  def transform(self, text):
    for tf in self.transformers:
      text = tf.transform(text)
    return text