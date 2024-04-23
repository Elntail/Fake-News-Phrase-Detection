#!/usr/bin/env python3
#     chmod u+x asmt0.py
#     ./asmt0.py

# Importing libraries 
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
import yake 


def generate_keyword_freq(text: str) -> list[tuple[str, int]] :
    ''' Generates a list of keywords through yake '''
    # Initializing the YAKE instance 
    yake_kw = yake.KeywordExtractor() 

    # Extracting keywords 
    KeyWords = yake_kw.extract_keywords(text) 
  
    return KeyWords

def only_keywords(KeyWords: list[tuple[str, int]]) -> list[str]:
    ''' Get only the keywords '''
    return [kw for kw, _ in KeyWords] 

def print_keywords(keywords: list[tuple[str, int]]):
    ''' Prints keywords and their score, 
        with the lowest scores first 
    '''

    keywords.sort(key = lambda a: a[1])
    for name,freq in keywords:
        print(f"\"{name}\" score: {(freq):>0.3f}\n")

def remove_single_words(KeyWords: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return [(kw,score) for (kw,score) in KeyWords if len(kw.split()) > 1] 



def main():
    # Input text 
    input_text = ''' 
    Hillary’s crime family: End of days for the U.S.A
    '''

    
    keywords = generate_keyword_freq(input_text)
    keywords = remove_single_words(keywords)

    print_keywords(keywords)    

    # Generate WordCloud 
    # wordcloud = WordCloud().generate(' '.join(keywords)) 
  
    # Display the WordCloud 
    # plt.figure(figsize=(10, 10)) 
    # plt.imshow(wordcloud, interpolation='bilinear') 
    # plt.axis('on') 
    # plt.show()

  

  

  


if __name__ == "__main__":
    main()
    
