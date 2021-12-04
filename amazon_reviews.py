import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

reviewlist = []


def get_soup(url):
    r = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try:
        for item in reviews:
            review = {
                'product': soup.title.text.replace('Amazon.co.uk:Customer reviews:', '').strip(),
                'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
                'rating': float(
                    item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
                'body': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            reviewlist.append(review)
    except:
        pass


for x in range(1, 11):
    soup = get_soup(
        f'https://www.amazon.com/AMD-Ryzen-5950X-32-Thread-Processor/product-reviews/B0815Y8J9N/ref=cm_cr_arp_d_paging_btm_next_2?pageNumber={x}')
    print(f'Getting page: {x}')
    get_reviews(soup)
    print(len(reviewlist))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break

df = pd.DataFrame(reviewlist)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1


df['sentiment'] = df['body'].apply(lambda x: sentiment_score(x[:46]))

df.to_excel('ryzen9-5950x.xlsx', index=False)
print("The End")



# #
# # def get_sentiment_score(stars, review):
# #     # Create a SentimentIntensityAnalyzer object.
# #     sid_obj = SentimentIntensityAnalyzer()
# #
# #     # polarity_scores method of SentimentIntensityAnalyzer
# #     # oject gives a sentiment dictionary.
# #     # which contains pos, neg, neu, and compound scores.
# #     sentiment_dict = sid_obj.polarity_scores(review)
# #
# #     print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
# #     print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
# #     print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")
# #     print("sentence has a compound score of ", sentiment_dict['compound'])
# #
# #     # decide sentiment as positive, negative and neutral
# #     if stars > 3 and sentiment_dict['compound'] >= 0.05:
# #         reviewRate.append("Positive review")
# #         print("Sentence Overall Rated As Positive")
# #
# #     elif stars < 3 and sentiment_dict['compound'] <= - 0.05:
# #         reviewRate.append("Negative review")
# #         print("Sentence Overall Rated As Negative")
# #
# #     elif (stars >=2 or stars <5) and sentiment_dict['compound'] >= - 0.05  and sentiment_dict['compound'] <=  0.05:
# #         reviewRate.append("Neutral review")
# #         print("Sentence Overall Rated As Neutral")
# #
# #     else:
# #         reviewRate.append("Undefined")
# #
# #
# # for row in df.itertuples():
# #     get_sentiment_score(float(row[3]), row[4])
# #
# # df.insert(len(df.columns), "Description of review", reviewRate)
# # print("Total count of reviews:",df["Description of review"].count())
# # print("Total count of positive reviews:", sum(df["Description of review"] == "Positive review"))
# # print("Total count of negative reviews:", sum(df["Description of review"] == "Negative review"))
# # print("Total count of neutral reviews:", sum(df["Description of review"] == "Neutral review"))
# # print("Total count of reviews that was undefined:", sum(df["Description of review"] == "Undefined"))
# # df.to_excel('ryzen9-5950x.xlsx', index=False)
# # print('Fine.')
#
#
#
# import numpy as np
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Embedding
# from keras.layers import LSTM, SpatialDropout1D
# from keras.datasets import imdb
#
# # Устанавливаем seed для повторяемости результатов
# np.random.seed(42)
# # Максимальное количество слов (по частоте использования)
# max_features = 5000
# # Максимальная длина рецензии в словах
# maxlen = 80
#
# # Загружаем данные
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
# print(imdb)
# # Заполняем или обрезаем рецензии
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#
# # Создаем сеть
# model = Sequential()
# # Слой для векторного представления слов
# model.add(Embedding(max_features, 32))
# model.add(SpatialDropout1D(0.2))
# # Слой долго-краткосрочной памяти
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# # Полносвязный слой
# model.add(Dense(1, activation="sigmoid"))
#
# # Копмилируем модель
# # model.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               metrics=['accuracy'])
# #
# # # Обучаем модель
# # model.fit(X_train, y_train, batch_size=64, epochs=7,
# #           validation_data=(X_test, y_test), verbose=2)
# # # Проверяем качество обучения на тестовых данных
# # scores = model.evaluate(X_test, y_test,
# #                         batch_size=64)
# # print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
#
#
#
#
#
#
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
#
# reviewlist = []
#
#
# def get_soup(url):
#     r = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
#     soup = BeautifulSoup(r.text, 'html.parser')
#     return soup
#
#
# def get_reviews(soup):
#     reviews = soup.find_all('div', {'data-hook': 'review'})
#     try:
#         for item in reviews:
#             review = {
#                 'product': soup.title.text.replace('Amazon.in:Customer reviews:', '').strip(),
#                 # 'product': soup.title.text.replace('Amazon.in:Customer reviews:', '').strip(),
#                 'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
#                 'rating': float(
#                     item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
#                 'body': item.find('span', {'data-hook': 'review-body'}).text.strip(),
#             }
#             reviewlist.append(review)
#     except:
#         pass
#
#
# for x in range(1, 5):
#     soup = get_soup(
#         f'https://www.amazon.in/Samsung-Galaxy-Storage-Additional-Exchange/product-reviews/B086KFBNV5/ref=cm_cr_getr_d_paging_btm_prev_2?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
#     print(f'Getting page: {x}')
#     get_reviews(soup)
#     print(len(reviewlist))
#     if not soup.find('li', {'class': 'a-disabled a-last'}):
#         pass
#     else:
#         break
# df = pd.DataFrame(reviewlist)


