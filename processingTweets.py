from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re

def process(df):
    # Removing numerals:
    df['paper_text_tokens'] = df.text.map(lambda x: re.sub(r'\d+', '', x))
    # Lower case:
    df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: x.lower())
    df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
    snowball = SnowballStemmer("english")  
    df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [snowball.stem(token) for token in x])
    stop_en = stopwords.words('english')
    df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if t not in stop_en]) 
    df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if len(t) > 2])
#    print(df['paper_text_tokens'][0][:500])