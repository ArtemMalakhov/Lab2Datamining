import csv  # read and write tabular data in CSV format
import sys
from tkinter import Tk, Label, Entry, Button

import controller as controller
import pandas as pd  # библиотека для анализа данных

from matplotlib import pyplot as plt  # для графиков
import re  # регулярные выражения
import nltk

from nltk.tokenize import word_tokenize  # для разбиения на предложения

nltk.download('punkt')  # загрузить все данные nltk
import numpy as np  # массивы

from tkinter.filedialog import askopenfilename

ham_list = list()
spam_list = list()

columns = ["v1", "v2"]  # надписи над ham/spam и соответствующими им сообщениями

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file

fileReader = pd.read_csv(filename, encoding='ISO-8859-1', usecols=columns)  # считываем из .csv файла
size = len(fileReader)

stopWords = ['a', 'in', 'to', 'the']

for i in range(size):
    string = fileReader.loc[i, 'v2']
    string = string.lower()
    string = re.sub(r'[^A-Za-z\s]+', '', string)
    fileReader.loc[i, 'v2'] = string  # возвращаем нормализованное слово
    tokens = word_tokenize(string)
    no_stopwords_list = [k for k in tokens if not k in stopWords]
    if str(fileReader.loc[i, 'v1']) == 'ham':
        ham_list.extend(no_stopwords_list)
    elif str(fileReader.loc[i, 'v1']) == 'spam':
        spam_list.extend(no_stopwords_list)
    string = ' '.join(no_stopwords_list)
    fileReader.loc[i, 'v2'] = string

fileReader.to_csv("update-sms-spam-corpus.csv", index=False)  # нормализированный массив предложений

count_ham_dict = dict()
for i in ham_list:
    count_ham_dict[i] = count_ham_dict.get(i, 0) + 1

count_spam_dict = dict()
for i in spam_list:
    count_spam_dict[i] = count_spam_dict.get(i, 0) + 1

words_ham = pd.DataFrame.from_dict(count_ham_dict, orient="index")
words_spam = pd.DataFrame.from_dict(count_spam_dict, orient="index")

words_ham.to_csv("ham_frequency_words.csv")
words_spam.to_csv("spam_frequency_words.csv")

with open('update-sms-spam-corpus.csv', 'r', encoding='ISO-8859-1') as read_file:
    csvReader = csv.reader(read_file)
    stopWords = ['a', 'in', 'to', 'the']
    myPorterStemmer = nltk.stem.porter.PorterStemmer()  # стемминг, выбираем основу слова
    str_arrays = []
    ham_array = {}
    spam_array = {}
    for line in csvReader:
        if line[0] == 'ham':
            for word in line:
                if word != '':
                    if word not in stopWords:
                        ham_array[myPorterStemmer.stem(word)] = ham_array.setdefault(myPorterStemmer.stem(word), 0) + 1
        else:
            for word in line:
                if word != '':
                    if word not in stopWords:
                        spam_array[myPorterStemmer.stem(word)] = spam_array.setdefault(myPorterStemmer.stem(word),
                                                                                       0) + 1

    ham_array = dict(sorted(ham_array.items(), key=lambda item: len(item[0]), reverse=True))
    str_arrays.append(ham_array)
    spam_array = dict(sorted(spam_array.items(), key=lambda item: len(item[0]), reverse=True))
    str_arrays.append(spam_array)

    with open('sms_all_words.csv', 'w', newline='') as dictionary:
        writer = csv.writer(dictionary)
        writer.writerow(['type', 'word', 'length'])

        for key in str_arrays[0].keys():
            writer.writerow(['ham', key, len(key)])

        for key in str_arrays[1].keys():
            writer.writerow(['spam', key, len(key)])

wordsAmount = 0
countWordsHam = 0
countWordsSpam = 0

with open('ham_frequency_words.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            wordsAmount += int(line[1].replace('\n', ''))
            countWordsHam += int(line[1].replace('\n', ''))

with open('spam_frequency_words.csv', 'r') as dictionary:
    reader = csv.reader(dictionary)
    for line in reader:
        if line[0] != 'word':
            wordsAmount += int(line[1].replace('\n', ''))
            countWordsSpam += int(line[1].replace('\n', ''))

pHam = countWordsHam / wordsAmount
pSpam = countWordsSpam / wordsAmount

all_words = pd.read_csv('sms_all_words.csv')
hams = all_words[all_words.type == 'ham']
spams = all_words[all_words.type == 'spam']

window = Tk()
window.title("Second lab")
window.configure(background="purple")
window.columnconfigure([0, 1, 2, 3, 4, 5, 6], minsize=500)
window.rowconfigure([0, 1, 2, 3, 4, 5, 6], minsize=50)
window.rowconfigure([6], minsize=150)
window.geometry('500x400')
window.resizable(width=False, height=False)
lbl = Label(window, text="Введите слово:", font=("Ghotam Pro", 14))
lbl.grid(column=0, row=0)

txt = Entry(window, width=20)
txt.grid(column=0, row=1)

lbl2 = Label(window, font=("Ghotam Pro", 14))
lbl3 = Label(window, font=("Ghotam Pro", 14))
lbl4 = Label(window, font=("Ghotam Pro", 14))


def clicked():
    spam_probability = 0
    ham_probability = 0
    num_in_ham = 0
    num_in_spam = 0

    res = txt.get()

    with open('ham_frequency_words.csv', 'r') as dictionary:
        reader = csv.reader(dictionary)
        for word in res:
            for line in reader:
                if line[0] == word:
                    num_in_ham += int(line[1])

    print(num_in_ham)
    with open('spam_frequency_words.csv', 'r') as dictionary:
        reader = csv.reader(dictionary)
        for word in res:
            for line in reader:
                if line[0] == word:
                    num_in_spam += int(line[1])
    print(num_in_spam)
    # for word in res:
    #     for line in ham_list:
    #         if line[0] != 'word':
    #             if line[0] == word:
    #                 num_in_ham = line[1]
    #
    # for word in res:
    #     for line in spam_list:
    #         if line[0] != 'word':
    #             if line[0] == word:
    #                 num_in_spam = line[1]

    spam_probability *= (num_in_spam + 1) / pSpam
    ham_probability *= (num_in_ham + 1) / pHam

    if spam_probability > ham_probability:
        lbl2.configure(text="Spam")
        lbl2.grid(column=0, row=3)
    else:
        lbl2.configure(text="Ham")
        lbl2.grid(column=0, row=3)

    lbl3.configure(text="Spam: %d" % spam_probability)
    lbl4.configure(text="Ham: %d" % ham_probability)
    lbl3.grid(column=0, row=4)
    lbl4.grid(column=0, row=5)


btn = Button(window, text="Поиск", command=clicked)
btn.grid(column=0, row=2)


def quit_program():
    window.destroy()
    sys.exit()


button3 = Button(window, text="Выход", command=quit_program)
button3.grid(column=0, row=6)

window.mainloop()
