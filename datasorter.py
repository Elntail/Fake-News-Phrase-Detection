#!/usr/bin/env python3

import csv
import pandas as pd
import sys
import os
from langdetect import detect


def split_data():
    file = '/home/rlantajo/Desktop/CS366/WELFake_Dataset.csv'
    # Test with test data

    reader = csv.reader(open(file))
    writer = csv.writer(open("dataset.csv", 'w'), delimiter='|')
    writer.writerows(reader)

    
    data = pd.read_csv('dataset.csv', delimiter='|')

    # Split data by label
    gk = data.groupby('label')
    real = gk.get_group(0).sample(frac=1).reset_index(drop=True)
    fake = gk.get_group(1).sample(frac=1).reset_index(drop=True)


    print('real before: ', len(real))
    print('fake before: ', len(fake))

    # Save data into seperate files
    real.to_csv('real_data.csv', sep='|')
    fake.to_csv('fake_data.csv', sep='|')

    
    real = pd.read_csv('real_data.csv', sep='|')
    fake = pd.read_csv('fake_data.csv', sep='|')

    print('real after: ', len(real))
    print('fake after: ', len(fake))

def clean_data():
    with open('real_data.csv', 'r') as real, open('fake_data.csv', 'r') as fake:
        real_reader = csv.reader(real, delimiter='|')
        fake_reader = csv.reader(fake, delimiter='|')

        with open('Cleaned_Real_Dataset.csv', 'w') as file:
            writer = csv.writer(file, delimiter='|')

            rows = []

            for row in real_reader:
                # row[2] = title
                # row[3] = text
                if row[2].strip() == '' or row[3].strip() == '':
                    continue
                
                title = row[2].replace('\n', '')
                text = row[3]. replace('\n', '')
                rows.append([title, text, row[4]])  
            
            print('real: ', len(rows))
            # print(rows[1])

            writer.writerows(rows)


        with open('Cleaned_Fake_Dataset.csv', 'w') as file:
            writer = csv.writer(file, delimiter='|')

            rows = []

            for row in fake_reader:
                # row[2] = title
                # row[3] = text
                if row[2].strip() == '' or row[3].strip() == '':
                    continue

                title = row[2].replace('\n', '')
                text = row[3]. replace('\n', '')
                rows.append([title, text, row[4]])    

            print('fake: ', len(rows))
            # print(rows[1])

            writer.writerows(rows)
    
    # real = pd.read_csv('Cleaned_Real_Dataset.csv', sep='|')
    # fake = pd.read_csv('Cleaned_Fake_Dataset.csv', sep='|')

    # real.drop(real.columns[[0,1]], inplace=True, axis=1)
    # real.to_csv('Cleaned_Real_Dataset.csv', sep='|')

    # fake.drop(fake.columns[[0,1]], inplace=True, axis=1)
    # fake.to_csv('Cleaned_Fake_Dataset.csv', sep='|')

def combine_data():
    real = pd.read_csv('Cleaned_Real_Dataset.csv', sep='|')
    fake = pd.read_csv('Cleaned_Fake_Dataset.csv', sep='|')


    total_data = pd.concat([real, fake])

    print('total: ', len(total_data))

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
    train = pd.concat([real_train, fake_train]).reset_index(drop=True)
    test = pd.concat([real_test, fake_test]).reset_index(drop=True)

    print('train: ', len(train))
    print('test: ', len(test))

    # Create CSV
    train.to_csv('train.csv', sep='|')
    test.to_csv('test.csv', sep='|')


    

def main():
    csv.field_size_limit(sys.maxsize)
    split_data()

    clean_data()
    
    combine_data()


if __name__ == "__main__":
    main()
    