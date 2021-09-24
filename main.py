import csv

import pandas as pd
import pylab

from matplotlib import pyplot as plt
import re
import nltk.corpus
from nltk.tokenize import word_tokenize

nltk.download('punkt')
import numpy as np

final_stopwords_list = ['a', 'in', 'the', 'to']

col_list = ["v1", "v2"]
sample = pd.read_csv("sms-spam-corpus.csv", encoding='ISO-8859-1', usecols=col_list)
size = len(sample)

list_for_ham = list()
list_for_spam = list()

for i in range(size):
    string = sample.loc[i, 'v2']
    string = string.lower()
    string = re.sub(r'[^A-Za-z\s]+', '', string)
    sample.loc[i, 'v2'] = string
    tokens = word_tokenize(string)
    list_without_stopwords = [k for k in tokens if not k in final_stopwords_list]
    if str(sample.loc[i, 'v1']) == 'ham':
        list_for_ham.extend(list_without_stopwords)
    elif str(sample.loc[i, 'v1']) == 'spam':
        list_for_spam.extend(list_without_stopwords)
    string = ' '.join(list_without_stopwords)
    sample.loc[i, 'v2'] = string

sample.to_csv("output/update_sms-spam-corpus.csv", index=False)

count_ham_dict = dict()
for i in list_for_ham:
    count_ham_dict[i] = count_ham_dict.get(i, 0) + 1

count_spam_dict = dict()
for i in list_for_spam:
    count_spam_dict[i] = count_spam_dict.get(i, 0) + 1

words_ham = pd.DataFrame.from_dict(count_ham_dict, orient="index")
words_spam = pd.DataFrame.from_dict(count_spam_dict, orient="index")

words_ham.to_csv("output/ham_counting_words.csv")
words_spam.to_csv("output/spam_counting_words.csv")

update_count_ham_dict = dict(sorted(count_ham_dict.items(), key=lambda x: x[1], reverse=True)[:20])
labels, values = zip(*update_count_ham_dict.items())
indexes = np.arange(len(labels))
plt.rcParams["figure.autolayout"] = True
plt.bar(indexes, values, align='center')
plt.xticks(indexes + 0.5, labels, rotation=90)
plt.savefig('output/ham_repeated_words')
plt.show()
plt.clf()

update_count_spam_dict = dict(sorted(count_spam_dict.items(), key=lambda x: x[1], reverse=True)[:20])
labels, values = zip(*update_count_spam_dict.items())
indexes = np.arange(len(labels))
plt.rcParams["figure.autolayout"] = True
plt.bar(indexes, values, align='center')
plt.xticks(indexes + 0.5, labels, rotation=90)
plt.savefig('output/spam_repeated_words')
plt.show()
plt.clf()

# Calculating total average length of all words

average_length = 0
total_sum_length = 0
words_amount = 0

with open('output/spam_counting_words.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            total_sum_length += len(line[0]) * int(line[1].replace('\n', ''))
            words_amount += int(line[1].replace('\n', ''))

with open('output/ham_counting_words.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            total_sum_length += len(line[0]) * int(line[1].replace('\n', ''))
            words_amount += int(line[1].replace('\n', ''))

average_length = total_sum_length / words_amount

# Creating plot with normalized words lengths and average length
hams = pd.read_csv('output/ham_counting_words.csv', encoding='ISO-8859-1', usecols=col_list)
spams = pd.read_csv('output/spam_counting_words.csv', encoding='ISO-8859-1', usecols=col_list)

pylab.subplot(2, 2, 1)

pylab.plot(range(len(hams)), hams.length / total_sum_length)
pylab.plot(range(len(spams)), spams.length / total_sum_length)
pylab.plot(range(len(hams)), hams.length * 0 + average_length / total_sum_length)
pylab.title('Words lengths')
pylab.ylabel("length")
pylab.legend(['ham', 'spam', 'average'])

# Getting ham and spam messages
# and creating arrays of normalized messages lengths
# and calculating average length of all messages

all_messages = pd.read_csv('sms-spam-corpus.csv', encoding='ISO-8859-1', usecols=col_list)
ham_messages = all_messages[all_messages.v1 == 'ham']
spam_messages = all_messages[all_messages.v1 == 'spam']
ham_messages_lengths = [len(m) for m in ham_messages.v2]
spam_messages_lengths = [len(m) for m in spam_messages.v2]
ham_messages_lengths = sorted(ham_messages_lengths, reverse=True)
spam_messages_lengths = sorted(spam_messages_lengths, reverse=True)
average_message_length = 0
total_messages_length = 0
number_of_messages = len(ham_messages) + len(spam_messages)

for length in ham_messages_lengths:
    total_messages_length += length

for length in spam_messages_lengths:
    total_messages_length += length

for i, length in enumerate(ham_messages_lengths):
    ham_messages_lengths[i] = length / total_messages_length

for i, length in enumerate(spam_messages_lengths):
    spam_messages_lengths[i] = length / total_messages_length

average_message_length = total_messages_length / number_of_messages

# Creating plot with normalized messages lengths and average length

x = np.linspace(0, len(ham_messages), len(ham_messages))

pylab.subplot(2, 2, 2)
pylab.plot(range(len(ham_messages)), ham_messages_lengths)
pylab.plot(range(len(spam_messages)), spam_messages_lengths)
pylab.plot(x, x * 0 + average_message_length / total_messages_length)
pylab.title('Messages lengths')
pylab.ylabel("length")
pylab.legend(['ham', 'spam', 'average'])

ham_dictionary = pd.read_csv('sms-ham-dictionary.csv', header=None, index_col=0, squeeze=True).to_dict()
ham_dictionary = sorted(ham_dictionary.items(), key=lambda item: int(item[1]), reverse=True)
most_frequent_ham_words = list(ham_dictionary)[:20]
most_frequent_ham_words_dict = {x[0]: x[1] for x in most_frequent_ham_words}
most_frequent_ham_words_dict = dict(sorted(most_frequent_ham_words_dict.items(), key=lambda item: item[1]))