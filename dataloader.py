#!/usr/bin/env python3

import csv
import pandas as pd
import sys
import os
import re
from langdetect import detect
csv.field_size_limit(sys.maxsize)



def split_data():
    ''' 
    Takes a single CSV with delim = ',' to use delim '|' for processing
    with the name 'dataset.csv'/. Then separate the dataset into its
    two label classifications under 'real_data.csv' and 'fake_data.csv' to
    balance later
    '''
    # Replace with pathway of file
    file = '/home/rlantajo/Desktop/CS366/WELFake_Dataset.csv'
    
    # Update given file to use the delimiter | for processing
    reader = csv.reader(open(file))
    writer = csv.writer(open("./data/dataset.csv", 'w'), delimiter='|')
    writer.writerows(reader)

    
    data = pd.read_csv('./data/dataset.csv', delimiter='|')

    # Split data by label and randomize
    gk = data.groupby('label')
    real = gk.get_group(0).sample(frac=1).reset_index(drop=True)
    fake = gk.get_group(1).sample(frac=1).reset_index(drop=True)


    print('real before: ', len(real))
    print('fake before: ', len(fake))

    # Save data into seperate files
    real.to_csv('./data/real_data.csv', sep='|')
    fake.to_csv('./data/fake_data.csv', sep='|')

    
    real = pd.read_csv('./data/real_data.csv', sep='|')
    fake = pd.read_csv('./data/fake_data.csv', sep='|')

    print('real after: ', len(real))
    print('fake after: ', len(fake))

def clean_string(string: str) -> str:
    ''' 
    Cleans a string of text that of unnecessary text

    :param string: full text or title from dataset
    :return: filtered string without undesired words and characters

    '''
    # Remove any paragraph breaks
    string = string.replace('\n', '')
    # Remove 1/11/11 or 11/11/11 dates
    string = re.sub(r'[1-9]+(/|-)[0-9][0-9](/|-)[0-9][0-9]', '', string)
    # Remove [word] or (word)
    string = re.sub(r'\[[A-Za-z0-9]+\]|\([A-Za-z0-9]+\)', '', string)
    # Remove links
    string = re.sub(r'https://[A-Za-z0-9]+(\.[A-Za-z0-9]+)+(/[A-Za-z0-9]+((-|_)?[A-Za-z0-9]+))+(\.[A-Za-z0-9]+)?', '', string)
    # Replace special characters with blank space
    string = re.sub(r'…|–', ' ', string)
    # Replace trailing dots with a single dot
    string = re.sub(r'\.\.\.', '.', string)
    # Remove hashtags
    string = re.sub(r'#', '', string)
    string = re.sub(r'\|', '.', string)

    return str(string)


def create_csv(data, path:str, save_as:str, delim:str, includeText:bool):
    '''
    Create a dataset
    '''

    reader = csv.reader(data, delimiter=delim)
    
    with open(path+save_as, 'w') as file:
        writer = csv.writer(file, delimiter='|')

        rows = []

        for row in reader:
            # Formated as in WELFake dataset
            
            # Clean title
            title = row[2]
            cleaned_title = clean_string(title)
                
            # Remove any empty cells 
            if cleaned_title.strip() == '':
                continue
                
            # Do the same for text if turned on
            if includeText:
                text = row[3]
                cleaned_text = clean_string(text)   
                if cleaned_text.strip() == '':
                    continue
                rows.append([cleaned_title, cleaned_text, row[4]]) 
            else:
                # Do not include cleaned text if turned off
                rows.append([cleaned_title, row[4]])  
            
        print(save_as,':', len(rows))

        writer.writerows(rows)



def clean_data(includeText=False, delete_files=True):
    '''
    Remove empty cells and clean all text of odd text and ascii characters
    '''

    with open('./data/real_data.csv', 'r') as real, open('./data/fake_data.csv', 'r') as fake:
        real_reader = csv.reader(real, delimiter='|')
        
        path = './data/'
        file_name1 = 'Cleaned_Real_Dataset.csv'
        file_name2 = 'Cleaned_Fake_Dataset.csv'

        fake_reader = csv.reader(fake, delimiter='|')

        # Create and clean each datasets
        create_csv(real, path, file_name1, '|', includeText)
        create_csv(fake, path, file_name2, '|', includeText)

    # Deletes old data
    # Comment out if you want to keep them
    os.remove('./data/fake_data.csv')
    os.remove('./data/real_data.csv')
    os.remove('./data/dataset.csv')

def combine_data():
    '''
    Combine the two datasets with a 25-75 even split to generate
    an even training and testing datasets
    '''
    real = pd.read_csv('./data/Cleaned_Real_Dataset.csv', sep='|')
    fake = pd.read_csv('./data/Cleaned_Fake_Dataset.csv', sep='|')

    n_real = len(real)
    n_fake = len(fake)

    # print('real before:', n_real)
    # print('fake before:', n_fake)

    # Make both datasets even sizes
    if n_real > n_fake:
        remove = real.sample(n=(n_real-n_fake))
        real = real.drop(remove.index)
    elif n_real < n_fake:
        remove = fake.sample(n=(n_fake-n_real))
        fake = fake.drop(remove.index)
    
    # print('real after:', len(real))
    # print('fake fake:', len(fake))

    total_data = pd.concat([real, fake])

    # print('total: ', len(total_data))

    # Randomize the data again 
    real_randomized = real.sample(frac=1)
    fake_randomized = fake.sample(frac=1)

    # Get 75% of each real and fake datasets 
    real_train = real_randomized.sample(frac=.75)
    fake_train = fake_randomized.sample(frac=.75)
    
    # Get 25% of each real and fake datasets 
    real_test = real_randomized.drop(real_train.index)
    fake_test = fake_randomized.drop(fake_train.index)

    # Combine both training datasets and both test datasets 
    train = pd.concat([real_train, fake_train]).sample(frac=1).reset_index(drop=True)
    test = pd.concat([real_test, fake_test]).sample(frac=1).reset_index(drop=True)

    # print('train: ', len(train))
    # print('test: ', len(test))

    # Create CSV
    train.to_csv('./data/train.csv', sep='|')
    test.to_csv('./data/test.csv', sep='|')

    # Deletes old datasets
    # Comment out if you want to keep
    os.remove('./data/Cleaned_Real_Dataset.csv')
    os.remove('./data/Cleaned_Fake_Dataset.csv')


    

def main():
    split_data()

    # 
    clean_data()
    
    combine_data()


if __name__ == "__main__":
    main()
    