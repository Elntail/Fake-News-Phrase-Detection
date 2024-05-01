#!/usr/bin/env python3

import csv
import pandas as pd
import sys
import os
from langdetect import detect

def main():
    csv.field_size_limit(sys.maxsize)
    file = '/home/rlantajo/Desktop/CS366/WELFake_Dataset.csv'
    # Test with test data
    
    # data = pd.read_csv(file)

    # gk = data.groupby('label')
    # real = gk.get_group(0).sample(frac=1).reset_index(drop=True)
    # fake = gk.get_group(1).sample(frac=1).reset_index(drop=True)


    # print(len(real))
    # print(len(fake))

    # real.to_csv('real_data.csv')
    # fake.to_csv('fake_data.csv')
    
    real = pd.read_csv('real_data.csv')
    fake = pd.read_csv('fake_data.csv')

    print('real: ', len(real))
    print('fake: ', len(fake))

    
    with open('real_data.csv', 'r') as real, open('fake_data.csv', 'r') as fake:
        real_reader = csv.reader(real)
        next(real_reader, None)
        fake_reader = csv.reader(fake)
        next(fake_reader, None)
        header = ['', 'Unnamed: 0', 'title', 'text', 'label']

        modified_real = [header]
        modified_fake = [header]

        with open('Cleaned_Real_Dataset.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)

            rows = []

            for row in real_reader:
                if row[2].strip() == '' or row[3].strip() == '':
                    continue
                rows.append(row)
            
            print('real: ', len(rows))
            # print(rows[1])

            writer.writerows(rows)


        with open('Cleaned_Fake_Dataset.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)

            rows = []

            for row in fake_reader:
                if row[2].strip() == '' or row[3].strip() == '':
                    continue
                rows.append(row)    

            print('fake: ', len(rows))
            # print(rows[1])

            writer.writerows(rows)



if __name__ == "__main__":
    main()
    