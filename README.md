# Fake-News-Phrase-Detection Analysis Program

This program was created by Rommel Lantajo II and Riik Acharya of Vassar College for a final research paper for Professor Jonathan's Gordan's Computational Linguistics (CS366) to produce a training and testing csv from the dataset WELFake for Fake News Classification from Kaggle to both train and test a DistilBert transformer model on an article’s title classification through a simplification of the title single key phrase and/or the each sentence key phrases from YAKE (Yet Another Keyword Extractor).

## Rational
Fake News is a prevalent problem plaguing our society today. Especially with the rise of social media, it’s easier than ever to spread fake news to a wide variety of unsuspecting viewers. Not only that, but because of social media, it is easier for people to retreat into their own bubbles and only consume news that they know already backs up their own viewpoints. There had also been a rise of people not reading the actual article but instead only the title itself and make their own conclusions from it. But can key phrase(s) within the article that is can be used as an indication that the article is authentic or fake? 

## How to run the program?
### Install all relevant libraries:
* pip3 install git+https://github.com/LIAAD/yake
* pip3 install pandas
* pip3 install transformers
* pip3 install numpy
* pip3 install matplotlib

### Cleaning the dataset
#### Preparation:
Download all the files in this github repo and download the WELFake dataset (https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). You can use other datasets but you need to change how the program reads the column headers in both the dataloader.py and yake_implementation.py accordingly. Then within the dataloader.py, go to the splitdata() function and set the file variable to your dataset's pathway.
#### Processing the data:
To process the data, do the following commands in the terminal within the downloaded Fake-News-Phrase-Detection folder:
1) python3 dataloader.py
2) python3 yake_implementation.py
#### What does these programs do?
dataloader.py will create a folder called data that will contain two files called train.csv and test.csv. The train.csv will be used to train the distilbert model on the classfication while the test.csv will be used to produce a similar dataset in the yake_implementation.py and the testing of the distilbert model.

yake_implementation.py takes in the testing.csv and produces a file called yake_key_phrases.csv in a new folder called results. This file has the closest keyphrase predicted by the YAKE model for the title alongside the original score and the yake score. 

### Running the dataset into DistilBert
#### Preparation:
Create a new folder called ckpt to store the checkpoint files
#### DistilBert:
To run the distilbert model simply type in the terminal:
python3 distilBert.py
#### What does these programs do?

The data is first passed into a DistillBERT layer, and then into a neural network. The model first trains on the train.csv and then makes predictions on both the yake_key_phrases.csv and the test.csv, as well as train.csv itself. The reason why we test both test.csv and train.csv is because we can examine the performace of the classification of both files and testing the accuracy of the model overall. 

The neural network has 4 hidden layers. The program trains on a certain number of epochs, meaning it runs over and over again a certain number of times hoping to achieve better results each time. This number controlled by the variable NUM_EPOCHS. It is set to 20, but it may be changed to be any number without affecrting the neural network itself. However, the average test loss may start increasing after a while if there are too many epochs. After each epoch, the checkpoint folder saves the results of that epoch, so if anything were to happen during the program's run, if it is run again, the program will start from after the last checkpoint instead of all over again.

After the completion of all the epochs, the program computes the f1-score for the testing data and the yake phrases. The f1 score is a balance of both precision and recall.

### (Optional) Data visualization
#### Preparation:
None
#### matlab.py:
Simply run python3 matlab.py to get the charts.
#### What does the program do?
It takes both the yake_key_phrases.csv and distilbert-processed file yake_true_false_stats.csv and produce a new csv called results.csv that contain all previous headers and two new columns: phrase length and yake score. Then it would process the data with different columns. More detailed information is in the mathlab.py itself.


## References 
### In-depth journal paper at Information Sciences Journal

Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509, pp 257-289. pdf

### ECIR'18 Best Short Paper

Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). A Text Feature Based Automatic Keyword Extraction Method for Single Documents. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772, pp. 684 - 691. pdf

Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A. (2018). YAKE! Collection-independent Automatic Keyword Extractor. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in Computer Science, vol 10772, pp. 806 - 810. pdf

### Distilbert, a distilled version of bert: Smaller, faster, cheaper and lighter
Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: Smaller, faster, cheaper and lighter. ArXiv.org.
