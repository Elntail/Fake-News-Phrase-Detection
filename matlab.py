#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt 
import numpy as np
import csv 
import pandas as pd
csv.field_size_limit(sys.maxsize)


def add_data():
    ''' Combines the orignal dataset to the distilbert label'''
    yake_orig = 'results/yake_key_phrases.csv' 
    yake_distb = 'results/yake_true_false_stats.csv'

    
    orig = pd.read_csv(yake_orig, sep='|')
    distb = pd.read_csv(yake_distb, sep=',')

    # Get all phrase lengths
    phrase_len = distb['phrase'].apply(lambda x: len(x.split()))


    distb.insert(1, 'phrase length', phrase_len, allow_duplicates=True)
    distb.insert(5, 'yake score', orig['score'], allow_duplicates=True)
    distb.to_csv('results/results.csv')

def stacked_bargraph():
    ''' Creates a stack bargraph of sums of classification'''

    results = pd.read_csv('results/results.csv')
    gk = results.groupby('gold')

    

    real = [len(gk.get_group('t_real')), len(gk.get_group('t_fake'))]
    fake = [len(gk.get_group('f_real')), len(gk.get_group('f_fake'))]
    print(real, '\n', fake)
    
    index =['Real News', 'Fake News']
    bargraph = pd.DataFrame({'Correctly Labeled': real, 'Incorrectly Label':fake}, index=index)

    bargraph.plot.bar(title='Accuracy in Label Prediction', rot=0, stacked=True, color={'Correctly Labeled': 'green', 'Incorrectly Label':'red'})
    plt.show()





def graph():
    ''' Creates a graph of the relationship between percentage
        correct and the length of phrases
    '''

    results = pd.read_csv('results/results.csv')
    gk = results.groupby('phrase length')
    

    index = list(gk.groups.keys())

    accuracy = []
    for length in index:

        current_group = gk.get_group(length)
        # print(current_group)
        gold_group = current_group.groupby('gold')

        # print(list(gold_group.groups.keys()))
        total = len(current_group)
        # print(total)
        t_real = 0
        try:
            t_real =  len(gold_group.get_group('t_real'))
        except:
            t_real = 0

        t_neg= 0
        try:
            t_neg =  len(gold_group.get_group('t_fake'))
        except:
            t_neg = 0


        t_neg = len(gold_group.get_group('t_fake'))
        result = (t_real + t_neg) / total

        accuracy.append(result)
    
    # print(index)
    # print(accuracy)
    
    line_graph = pd.DataFrame({'phrase length': index, 'accuracy': accuracy})

    line_graph.plot.line(title='Key Phrase Length vs Accuracy', x='phrase length', y='accuracy', ylabel='Accuracy(%)', xticks=index)
    plt.show()

def scatterplot():
    ''' Produce a scatterplot that compares the given YAKE score to the Phrase length'''
    results = pd.read_csv('results/results.csv')
    phrases_length = results['phrase length'].tolist()
    yake_score = results['yake score'].tolist()


    gk = results.groupby('phrase length')
    

    index = list(gk.groups.keys())
    

    scatterplot = pd.DataFrame({'yake score': yake_score, 'phrase length': phrases_length})
    scatterplot.plot.scatter(title='Phrase Length vs Yake Score', x='phrase length', y='yake score', logy=True,  xticks=index, s=5, color='red')

    plt.show()

def density():
    ''' Compares density of yake scores'''
    results = pd.read_csv('results/results.csv')
    yake_score = results['yake score'].tolist()

    s = pd.Series(yake_score)
    s.plot.kde(title='Yake Score Density', logx=True, color='red')
    plt.show()



def main():

    add_data()
    stacked_bargraph()
    graph()
    scatterplot()
    density()


  

  

  


if __name__ == "__main__":
    main()
    
