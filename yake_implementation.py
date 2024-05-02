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
    language = "en"
    max_ngram_size = 5
    deduplication_threshold = .9
    deduplication_algo = 'seqm'
    windowSize = 20
    numOfKeywords = 1
    
    
    
    
    
    
    # Initializing the YAKE instance 
    # yake_kw = yake.KeywordExtractor() 
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


def generate_keywords(story: str) -> list[list[str, int]]:
    # print(state, ' news:')
    
    sentences = get_sentences(story)


    list_keyphrases = []
    for sentence in sentences:
        keywords = generate_keyword_freq(sentence)
        keywords = remove_single_words(keywords)
        
        list_keyphrases.append(keywords)
        
        # print_keywords(keywords)
    # print()

    return list_keyphrases    


def generate_csv(includeText = False):

    # Only include test.csv as yake works as an unsupervised program
    with open('test.csv', 'r') as test:

        test_reader = csv.reader(test, delimiter='|')
        # Skip header row
        next(test_reader)

        with open('yake_key_phrases.csv', 'w') as file:
            writer = csv.writer(file, delimiter='|')

            rows = [['Phrase', 'score', 'label']]

            for row in test_reader:
                title = row[1]
                text = row[2]
                label = row[3]

                title_keyphrases = generate_keywords(title)
                for keyphrases in title_keyphrases:
                    for (keyphrase, score) in keyphrases:
                        rows.append([keyphrase, score, label])
                

                if includeText:
                    text_keyphrases = generate_keywords(text)
                    for keyphrases in text_keyphrases:
                        for (keyphrase, score) in keyphrases:
                            rows.append([keyphrase, score, label])
            
    
            writer.writerows(rows)




# Obtained from class
def get_sentences(text: str) -> list[str]:
    """Split the specified text into sentences, consisting of text tokens."""
    sents = []

    # A modified version of get_sentences from asmt 4 without
    # tokenizes the sentence itself
    doc = nlp(text)
    for sent in doc.sents:
        tokens = [token.text.lower().strip() for token in sent if not token.is_space]
        sents.append(' '.join(tokens))

    return sents


def main():
    csv.field_size_limit(sys.maxsize)


    # generate_csv()

    test = pd.read_csv('yake_key_phrases.csv', sep='|')
    print(len(test))

    test2 = pd.read_csv('test.csv', sep='|')
    print(len(test2))
    
    # Fake News Test
    # input_text = ''' 
    # STUDENTS SENT HOME From School For Wearing Traditional Swiss Clothing Considered “racist”
    # '''

    # fake_sen = get_sentences(input_text)
    # # print(fake_sen)
    # keywords = generate_keywords(fake_sen)


    # # Real News Test
    # input_text2 = '''
    # HANOI (Reuters) - At least 54 people died and 39 went missing as destructive floods battered northern and central Vietnam this week, the disaster prevention agency said on Friday.  Vietnam is prone to destructive storms and flooding due to its long coastline. A typhoon wrecked havoc across central provinces just last month.  The floods that hit Vietnam this week starting on Monday are the worst in years, state-run Vietnam Television quoted agriculture minister Nguyen Xuan Cuong as saying.  Nineteen people from four neighboring households in Hoa Binh were buried alive early on Thursday after a landslide struck around midnight on Wednesday, but only nine bodies have been found, the disaster agency said in a report.  Some 317 homes have collapsed in floods and landslides this week, while more than 34,000 other houses have been submerged or  damaged.  More than 22,000 hectares (54,300 acres) of rice have also been damaged and around 180,000 animals killed or washed away.    Floods have also affected seven of 77 provinces in Thailand, Vietnam s neighbor to the west, that country s Department of Disaster Prevention and Mitigation said on Thursday.  More than 480,000 hectares (1.2 million acres) of agricultural land Thailand have been hit, the department said.      
    # '''
    # real_sen = get_sentences(input_text2)
    # keyword2 = generate_keywords(real_sen)


    # test = '''
    # Conta-me Histórias." Xutos inspiram projeto premiado. A plataforma "Conta-me Histórias" foi distinguida com o Prémio Arquivo.pt, atribuído a trabalhos inovadores de investigação ou aplicação de recursos preservados da Web, através dos serviços de pesquisa e acesso disponibilizados publicamente pelo Arquivo.pt . Nesta plataforma em desenvolvimento, o utilizador pode pesquisar sobre qualquer tema e ainda executar alguns exemplos predefinidos. Como forma de garantir a pluralidade e diversidade de fontes de informação, esta são utilizadas 24 fontes de notícias eletrónicas, incluindo a TSF. Uma versão experimental (beta) do "Conta-me Histórias" está disponível aqui.
    # '''

    

    # print_keywords(keywords3)    



  

  

  


if __name__ == "__main__":
    main()
    
