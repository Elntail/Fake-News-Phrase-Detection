#!/usr/bin/env python3
#     chmod u+x asmt0.py
#     ./asmt0.py

# Importing libraries 
import sys
import pandas as pd
import yake 
import csv
from spacy.lang.en import English
nlp = English(pipeline=[])
nlp.add_pipe("sentencizer")


def generate_keyword_freq(text: str) -> list[tuple[str, int]] :
    ''' Generates a list of keywords through yake '''
    max_ngram_size = 20
    windowSize = 20
    numOfKeywords = 1
     
    
    # Initializing the YAKE instance 
    yake_kw = yake.KeywordExtractor(n=max_ngram_size, windowsSize=windowSize, top=numOfKeywords) 

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



def generate_keywords_title(title: str) -> list[list[str, int]]:
    
    keywords = generate_keyword_freq(title)
    keywords = remove_single_words(keywords)
    
    return keywords


def generate_keywords_text(story: str) -> list[list[str, int]]:
    ''' Process each sentence in the article and get a single keyphrase '''
    sentences = get_sentences(story)

    list_keyphrases = []
    for sentence in sentences:
        keywords = generate_keyword_freq(sentence)
        keywords = remove_single_words(keywords)
        
        list_keyphrases.append(keywords)
        

    return list_keyphrases


def process_keysphrases(keyphrases: list[list[str, int]], label: int) -> list[[str, int, int]]:
    ''' Append a label to keyphrases '''
    rows = []
    
    for (phrase, score) in keyphrases:
        # If it is only titles, runtime is O(n) as there is one tuple per keyphrases
        rows.append([phrase, score, label])
    
    return rows


def generate_csv(includeText = False):
    ''' Generate a csv with the original idx in test.csv, the keyphrase, score, and orignal label'''
    # Only include test.csv as yake works as an unsupervised program
    
    rows_del_idx = [] 
    with open('test.csv', 'r') as test:

        test_reader = csv.reader(test, delimiter='|')
        # Skip header row
        next(test_reader)

        with open('yake_key_phrases.csv', 'w') as file:
            writer = csv.writer(file, delimiter='|')

            rows = [['Key Phrase', 'score', 'label']]

            for row in test_reader:
                orig_idx = row[0]
                title = row[1]
                text = row[2]
                label = row[3]

                # Only generate a single keyphrase per title
                title_keyphrases = generate_keywords_title(title)

                # If title produces only one word key phrase, then add to
                # acc to delete from test.csv and skip
                if title_keyphrases == []:
                    rows_del_idx.append(int(orig_idx))
                    continue

                
                rows += process_keysphrases(title_keyphrases, label)

                if includeText:

                    # Also generate a single keyphrase per sentence in text
                    text_keyphrases = generate_keywords_text(text)
                    for keyphrases in text_keyphrases:    
                        rows += process_keysphrases(keyphrases, label)

            writer.writerows(rows)

    # Update test.csv
    test = pd.read_csv('test.csv', sep='|')
    test = test.drop(rows_del_idx).reset_index()
    test.to_csv('test.csv', sep='|')



# Obtained from class
def get_sentences(text: str) -> list[str]:
    """Split the specified text into sentences, consisting of text tokens."""
    sents = []

    doc = nlp(text)
    for sent in doc.sents:
        tokens = [token.text.lower().strip() for token in sent if not token.is_space]
        sents.append(' '.join(tokens))

    return sents


def main():
    csv.field_size_limit(sys.maxsize)

    generate_csv()

    test = pd.read_csv('test.csv', sep='|')
    print('test: ', len(test))

    yake = pd.read_csv('yake_key_phrases.csv', sep='|')
    print('yake: ', len(yake))


  

  

  


if __name__ == "__main__":
    main()
    
