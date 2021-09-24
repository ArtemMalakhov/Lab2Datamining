import csv

import nltk
import pandas as pd
import pylab

from matplotlib import pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

from nltk.stem import porter
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

sample.to_csv("update_sms-spam-corpus.csv", index=False)

count_ham_dict = dict()
for i in list_for_ham:
    count_ham_dict[i] = count_ham_dict.get(i, 0) + 1

count_spam_dict = dict()
for i in list_for_spam:
    count_spam_dict[i] = count_spam_dict.get(i, 0) + 1

words_ham = pd.DataFrame.from_dict(count_ham_dict, orient="index")
words_spam = pd.DataFrame.from_dict(count_spam_dict, orient="index")

words_ham.to_csv("ham_counting_words.csv")

words_spam.to_csv("spam_counting_words.csv")

with open('sms-spam-corpus.csv', 'r', encoding='ISO-8859-1') as read_file:
    csv_reader = csv.reader(read_file)
    spec_chars = "[^a-zA-Z ]"
    stop_words = ['a', 'in', 'the', 'to']
    myPorterStemmer = nltk.stem.porter.PorterStemmer()
    str_arrays = []
    ham_array = {}
    spam_array = {}
    for line in csv_reader:
        if line[0] == 'ham':
            for word in re.sub(spec_chars, '', line[1]).lower().split(" "):
                if word != '':
                    if word not in stop_words:
                        ham_array[myPorterStemmer.stem(word)] = ham_array.setdefault(myPorterStemmer.stem(word), 0) + 1
        else:
            for word in re.sub(spec_chars, '', line[1]).lower().split(" "):
                if word != '':
                    if word not in stop_words:
                        spam_array[myPorterStemmer.stem(word)] = spam_array.setdefault(myPorterStemmer.stem(word),
                                                                                       0) + 1

    ham_array = dict(sorted(ham_array.items(), key=lambda item: len(item[0]), reverse=True))
    spam_array = dict(sorted(spam_array.items(), key=lambda item: len(item[0]), reverse=True))
    str_arrays.append(ham_array)
    str_arrays.append(spam_array)

    with open('sms-all-words.csv', 'w', newline='') as dictionary:
        writer = csv.writer(dictionary)
        writer.writerow(['type', 'word', 'length'])

        for key in str_arrays[0].keys():
            writer.writerow(['ham', key, len(key)])

        for key in str_arrays[1].keys():
            writer.writerow(['spam', key, len(key)])

    with open('sms-ham-dictionary.csv', 'w', newline='') as dictionary:
        writer = csv.writer(dictionary)
        writer.writerow(['word', 'frequency'])

        for key, value in ham_array.items():
            writer.writerow([key, value])

    with open('sms-spam-dictionary.csv', 'w', newline='') as dictionary:
        writer = csv.writer(dictionary)
        writer.writerow(['word', 'frequency'])

        for key, value in spam_array.items():
            writer.writerow([key, value])

# Calculating total average length of all words

average_length = 0
total_sum_length = 0
words_amount = 0

with open('sms-spam-dictionary.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            total_sum_length += len(line[0]) * int(line[1].replace('\n', ''))
            words_amount += int(line[1].replace('\n', ''))

with open('sms-ham-dictionary.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            total_sum_length += len(line[0]) * int(line[1].replace('\n', ''))
            words_amount += int(line[1].replace('\n', ''))

average_length = total_sum_length / words_amount

# Creating plot with normalized words lengths and average length

all_words = pd.read_csv('sms-all-words.csv')
hams = all_words[all_words.type == 'ham']
spams = all_words[all_words.type == 'spam']

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

with open('sms-spam-dictionary.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            total_sum_length += len(line[0]) * int(line[1].replace('\n', ''))
            words_amount += int(line[1].replace('\n', ''))

with open('sms-ham-dictionary.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            total_sum_length += len(line[0]) * int(line[1].replace('\n', ''))
            words_amount += int(line[1].replace('\n', ''))

average_length = total_sum_length / words_amount

# Creating plot with normalized words lengths and average length

all_words = pd.read_csv('sms-all-words.csv')
hams = all_words[all_words.type == 'ham']
spams = all_words[all_words.type == 'spam']

pylab.plot(range(len(hams)), hams.length / total_sum_length)
pylab.plot(range(len(spams)), spams.length / total_sum_length)
pylab.plot(range(len(hams)), hams.length * 0 + average_length / total_sum_length)
pylab.title('Words lengths')
pylab.ylabel("length")
pylab.legend(['ham', 'spam', 'average'])

# Getting ham and spam messages
# and creating arrays of normalized messages lengths
# and calculating average length of all messages

all_messages = pd.read_csv('sms-spam-corpus.csv', encoding='ISO-8859-1')
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

plt.plot(range(len(hams)), hams.length / total_sum_length)
plt.plot(range(len(spams)), spams.length / total_sum_length)
plt.plot(range(len(hams)), hams.length * 0 + average_length / total_sum_length)
plt.title('Words lengths')
plt.ylabel("length")
plt.legend(['ham', 'spam', 'average'])
plt.savefig('output/1task_diagram')
plt.show()
plt.clf()

plt.plot(range(len(ham_messages)), ham_messages_lengths)
plt.plot(range(len(spam_messages)), spam_messages_lengths)
plt.plot(x, x * 0 + average_message_length / total_messages_length)
plt.title('Messages lengths')
plt.ylabel("length")
plt.legend(['ham', 'spam', 'average'])
plt.savefig('output/2task_diagram')
plt.show()
plt.clf()
