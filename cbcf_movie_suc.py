# coding: UTF-8
import csv
import numpy as np
import pandas as pd
import MeCab as mc
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
import itertools
from sklearn import metrics
import matplotlib.pyplot as plt

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def out_symbol(x):
    symbols = ["・", "，", ",", ".", "、", "。", "(", ")", "（", "）", "「", "」", "ー", "-", "*"]
    return x not in symbols

rating_data = pd.read_csv('ml-100k/u1.base', names=["user_id", "item_id", "rating", "timestamp"], sep="\t")
item_id = pd.read_csv('ml-100k/u.item', names=['item_id'], usecols=[0], sep='|', encoding="latin-1")
user_id = pd.read_csv('ml-100k/u.user', names=['user_id'], usecols=[0], sep='|', encoding="latin-1")

item_n = len(item_id)
user_n = len(user_id)
c_n = 5

content_name = deepcopy(item_id.values)

#testdataの読み込み
test_data = pd.read_csv('ml-100k/u1.test', names=["user_id", "item_id", "rating", "timestamp"], sep="\t")
one_rating_matrix_test = np.zeros((len(user_id), len(item_id)))
correct_value = np.zeros(len(test_data))
# テストデータの部分を１、その他を０とする
for i in tqdm(range(len(test_data))):
    one_rating_matrix_test[test_data.at[i, 'user_id']-1, test_data.at[i, 'item_id']-1] = 1
    correct_value[i] = test_data.at[i, 'rating']

test_data = deepcopy(test_data.values)

#user_id*item_idのtrain_dataを作成
train_data = np.zeros((user_n, item_n))
rating_cal = np.zeros((5, user_n, item_n))
one_rating_matrix = np.zeros((user_n, item_n))
for i in tqdm(range(len(rating_data))):
    train_data[rating_data.at[i, 'user_id']-1, rating_data.at[i, 'item_id']-1] = rating_data.at[i, 'rating']
    rating_cal[rating_data.at[i, 'rating']-1, rating_data.at[i, 'user_id']-1, rating_data.at[i, 'item_id']-1] = 1
    one_rating_matrix[rating_data.at[i, 'user_id']-1, rating_data.at[i, 'item_id']-1] = 1

#item_id*genreの行列を作る
item_content_data_cols =['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMBd_URL',
                        'unknown', 'Action', 'Adventure', 'Animation','Children', 'Comedy', 'Crime',
                        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
item_content_data = pd.read_csv('ml-100k/u.item', names=item_content_data_cols, sep='|', encoding="latin-1")
item_content_data = item_content_data.fillna(0)

#genre以外の列を削除
del item_content_data['movie_id']
del item_content_data['movie_title']
del item_content_data['release_date']
del item_content_data['video_release_date']
del item_content_data['IMBd_URL']
item_profile = deepcopy(item_content_data)
one_item_content_data = deepcopy(item_content_data)

#数字をジャンル名に置き換える
for i in range(len(item_content_data.columns)):
    item_content_data.loc[item_content_data[item_content_data.columns.values[i]]!=0, item_content_data.columns.values[i]]=item_content_data.columns.values[i]

# 各ユーザーが評価したコンテンツの数
n_user_rated = np.zeros(user_n)
for user in range(user_n):
    for item in range(item_n):
        if train_data[user][item] != 0:
            n_user_rated[user] += 1

words = [] # 全ての単語
content_words = [[] for i in range(item_n)] #コンテンツごとの単語
for i in tqdm(range(item_n)):
    for j in range(len(item_content_data.columns)):
        if item_content_data.iloc[i, j] != 0:
            content_words[i].append(str(item_content_data.iat[i, j]))
            words.append(str(item_content_data.iat[i, j]))

examples = pd.Series(words).value_counts()
vocabulary = pd.Series(words).unique()
words_size = len(words)
vocabulary_size = len(vocabulary)

words_user_rated = [[] for i in range(user_n)]
for user in range(user_n):
    # 評価したコンテンツの単語を入れていく
    for item in range(item_n):
        if train_data[user][item] != 0:
            words_user_rated[user].append(content_words[item])
    words_user_rated[user] = list(itertools.chain.from_iterable(words_user_rated[user]))
    words_user_rated[user] = pd.Series(words_user_rated[user])
    words_user_rated[user] = words_user_rated[user].unique()
    words_user_rated[user] = len(words_user_rated[user])

# 評価した単語の総数(重複あり)
tmp = np.zeros(user_n)
# tmp_w = np.zeros(user_n)
for i in range(user_n):
    for j in range(item_n):
        if train_data[i][j] != 0:
            tmp[i] += len(content_words[j])

p_c = np.zeros((user_n, c_n))
full_train_data = deepcopy(train_data)
for user in tqdm(range(user_n)):
    text = [[] for i in range(c_n)] # text[i]には、userが評価iとした単語
    docs = [[] for i in range(c_n)] # docs[i]には、userが評価iとした単語(重複なし)
    n_k = [[] for i in range(c_n)]
    #p_cの計算
    for c in range(c_n):
        for item in range(item_n):
            if train_data[user][item] == c + 1:
                text[c].append(content_words[item]) # userがcと評価した単語をまとめる
                p_c[user][c] += 1 # userがcと評価したコンテンツ数
        p_c[user][c] = (p_c[user][c] * words_size + 1) / (n_user_rated[user] * words_size + c_n) # userがコンテンツをcと評価する確率
        text[c] = list(itertools.chain.from_iterable(text[c]))
        text[c] = pd.Series(text[c])
        docs[c] = text[c].unique()
        n_k[c] = text[c].value_counts(sort = False)

    p_asc = np.zeros((c_n, vocabulary_size))
    for c in range(c_n):
        for i in range(len(docs[c])):
            # 評価cの中でのdocs[c][i]の割合
            p_asc[c][vocabulary.tolist().index(docs[c][i])] = (n_k[c][docs[c][i]] * words_size + 1) / (words_size * len(text[c]) + words_user_rated[user])

    for c in range(c_n):
        for i in range(vocabulary_size):
            if p_asc[c][i] == 0:
                p_asc[c][i] = 1.0 / (words_size * len(text[c]) + words_user_rated[user])

    for item in range(item_n):
        if full_train_data[user][item] == 0:
            p = [[] for i in range(c_n)]
            for c in range(c_n):
                p[c] = math.log10(p_c[user][c])
                for i in range(len(content_words[item])):
                    p[c] += math.log10(p_asc[c][vocabulary.tolist().index(content_words[item][i])])
            full_train_data[user][item] = np.argmax(p) + 1

cb_prediction_matrix = deepcopy(full_train_data)
for i in tqdm(range(user_n)):
    for j in range(item_n):
        if train_data[i][j] != 0:
            cb_prediction_matrix[i][j] = 0

cb_prediction = np.zeros(len(correct_value))
for i in range(len(correct_value)):
    cb_prediction[i] = full_train_data[int(test_data[i][0])-1][int(test_data[i][1])-1]

sw_max = 2

self_weight = (n_user_rated / 50) * sw_max
for i in range(user_n):
    if self_weight[i] > sw_max:
        self_weight[i] = sw_max

#ユーザーごとの、ratingの偏差を求める
rating_average = np.average(full_train_data, axis=1)
rate_deviation = deepcopy(full_train_data)
for i in range(item_n):
    rate_deviation[:,i] -= rating_average

#ユーザーの相関係数を求める
rate_deviation_sum = np.sum(rate_deviation, axis=1)
user_similarity = np.zeros((user_n, user_n))
for i in tqdm(range(user_n)):
    for j in range(user_n):
        if i != j:
            user_similarity[i][j] = cos_sim(full_train_data[i], full_train_data[j])

#上位30のユーザーをneighborhoodとする
neighborhood = np.argsort(user_similarity, axis=1)[:, ::-1][:, 1:31]

#neighborのhm
m = np.zeros(user_n)
for i in range(user_n):
    if n_user_rated[i] >= 50:
        m[i] = 1
    elif n_user_rated[i] < 50:
        m[i] = n_user_rated[i] / 50

hm = np.zeros((user_n, neighborhood.shape[1]))
for i in range(user_n):
    for j in range(neighborhood.shape[1]):
        hm[i][j] = 2 * m[i] * m[neighborhood[i][j]] / (m[i] + m[neighborhood[i][j]])

#neighborのsg
sg = np.zeros((user_n, neighborhood.shape[1]))

co_rated_item = np.zeros((user_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(neighborhood.shape[1]):
        for k in range(item_n):
            if train_data[i][k] != 0 and train_data[neighborhood[i][j]][k] != 0:
                co_rated_item[i][j] += 1
        if co_rated_item[i][j] >= 50:
            co_rated_item[i][j] = 50
sg = co_rated_item / 50

#neighborのhw
hw = hm + sg

#neighborの偏差と相関係数
corr_neighbor = np.zeros((user_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(neighborhood.shape[1]):
        corr_neighbor[i][j] = user_similarity[i][neighborhood[i][j]]

corr_deviation = np.zeros((user_n, item_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        for k in range(neighborhood.shape[1]):
            corr_deviation[i][j][k] = rate_deviation[neighborhood[i][k]][j]

#hw * p * (v-v)
hwpvv = deepcopy(corr_deviation)
for i in range(user_n):
    for j in range(item_n):
        hwpvv[i][j] *= corr_neighbor[i] * hw[i]
hwpvv_sum = np.zeros((user_n, item_n))

for i in range(user_n):
    tmp_4 = np.sum(hwpvv[i], axis=1)
    for j in range(item_n):
        hwpvv_sum[i][j] = tmp_4[j]

hwp = hw * corr_neighbor
hwp_sum = np.sum(hwp, axis=1)

#prediction
cbcf_prediction_matrix = np.zeros((user_n, item_n))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        a = self_weight[i] * (full_train_data[i][j] - rating_average[i]) + hwpvv_sum[i][j]
        b = self_weight[i] + hwp_sum[i]
        cbcf_prediction_matrix[i][j] = rating_average[i] + (a / b)
        # cbcf_prediction_matrix[i][j] = rating_average[i] + (np.sum(corr_deviation[i][j] * corr_neighbor[i]) / np.sum(corr_neighbor[i]))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        if train_data[i][j] != 0:
            cbcf_prediction_matrix[i][j] = 0

cbcf_prediction = np.zeros(len(correct_value))
for i in range(len(correct_value)):
    cbcf_prediction[i] = cbcf_prediction_matrix[int(test_data[i][0])-1][int(test_data[i][1])-1]

#MAE
cb_prediction = np.round(cb_prediction)
# cbcf_prediction = np.round(cbcf_prediction)

tmp_7 = np.abs(correct_value - cbcf_prediction)
cbcf_mae = np.sum(tmp_7) / len(correct_value)
tmp_7 = np.abs(correct_value - cb_prediction)
cb_mae = np.sum(tmp_7) / len(correct_value)

# cf
sw_max = 2
n_user_rated = np.zeros(user_n)

for user in range(user_n):
    for item in range(item_n):
        if train_data[user][item] != 0:
            n_user_rated[user] += 1

self_weight = (n_user_rated / 50) * sw_max
for i in range(user_n):
    if self_weight[i] > sw_max:
        self_weight[i] = sw_max

#ユーザーごとの、ratingの偏差を求める
rating_average = np.sum(train_data, axis=1) / np.sum(one_rating_matrix, axis=1)
rate_deviation = deepcopy(train_data)
for i in range(item_n):
    rate_deviation[:,i] -= rating_average

for i in range(user_n):
    for j in range(item_n):
        if train_data[i][j] == 0:
            rate_deviation[i][j] = 0

#ユーザーの相関係数を求める
rate_deviation_sum = np.sum(rate_deviation, axis=1)
user_similarity = np.zeros((user_n, user_n))
for i in range(user_n):
    for j in range(user_n):
        if i != j:
            user_similarity[i][j] = cos_sim(train_data[i], train_data[j])

#上位30のユーザーをneighborhoodとする
neighborhood = np.argsort(user_similarity, axis=1)[:, ::-1][:, 1:31]

#neighborのhm
m = np.zeros(user_n)
for i in range(user_n):
    if n_user_rated[i] >= 50:
        m[i] = 1
    elif n_user_rated[i] < 50:
        m[i] = n_user_rated[i] / 50

hm = np.zeros((user_n, neighborhood.shape[1]))
for i in range(user_n):
    for j in range(neighborhood.shape[1]):
        hm[i][j] = 2 * m[i] * m[neighborhood[i][j]] / (m[i] + m[neighborhood[i][j]])

#neighborのsg
sg = np.zeros((user_n, neighborhood.shape[1]))

co_rated_item = np.zeros((user_n, neighborhood.shape[1]))
for i in range(user_n):
    for j in range(neighborhood.shape[1]):
        for k in range(item_n):
            if train_data[i][k] != 0 and train_data[neighborhood[i][j]][k] != 0:
                co_rated_item[i][j] += 1
        if co_rated_item[i][j] >= 50:
            co_rated_item[i][j] = 50
sg = co_rated_item / 50

#neighborのhw
hw = hm + sg

#neighborの偏差と相関係数
corr_neighbor = np.zeros((user_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(neighborhood.shape[1]):
        corr_neighbor[i][j] = user_similarity[i][neighborhood[i][j]]

corr_deviation = np.zeros((user_n, item_n, neighborhood.shape[1]))
for i in tqdm(range(user_n)):
    for j in range(item_n):
        for k in range(neighborhood.shape[1]):
            corr_deviation[i][j][k] = rate_deviation[neighborhood[i][k]][j]

#hw * p * (v-v)
hwpvv = deepcopy(corr_deviation)
for i in range(user_n):
    for j in range(item_n):
        hwpvv[i][j] *= corr_neighbor[i] * hw[i]
hwpvv_sum = np.zeros((user_n, item_n))

for i in range(user_n):
    tmp_4 = np.sum(hwpvv[i], axis=1)
    for j in range(item_n):
        hwpvv_sum[i][j] = tmp_4[j]

hwp = hw * corr_neighbor
hwp_sum = np.sum(hwp, axis=1)

#prediction
cf_prediction_matrix = deepcopy(train_data)
for i in range(user_n):
    for j in range(item_n):
        if train_data[i][j] == 0:
            a = hwpvv_sum[i][j]
            b = hwp_sum[i]
            cf_prediction_matrix[i][j] = rating_average[i] + (a / b)
            # cf_prediction_matrix[i][j] = rating_average[i] + (np.sum(corr_deviation[i][j] * corr_neighbor[i]) / np.sum(corr_neighbor[i]))
for i in range(user_n):
    for j in range(item_n):
        if train_data[i][j] != 0:
            cf_prediction_matrix[i][j] = 0

cf_prediction = np.zeros(len(correct_value))
for i in range(len(correct_value)):
    cf_prediction[i] = cf_prediction_matrix[int(test_data[i][0])-1][int(test_data[i][1])-1]

#MAE
# cf_prediction = np.round(cf_prediction)
tmp_7 = np.abs(correct_value - cf_prediction)
cf_mae = np.sum(tmp_7) / len(correct_value)
print("cfの誤差:", cf_mae)
print("cbの誤差:", cb_mae)
print("cbcfの誤差:", cbcf_mae)
print()
print("cfの評価予測:", cf_prediction)
print("cbの評価予測:", cb_prediction)
print("cbcfの評価予測:", cbcf_prediction)
print("実際の評価:", correct_value)

cf_reclist = np.argsort(cf_prediction_matrix, axis=1)[:, ::-1][:, 1:31]
cb_reclist = np.argsort(cb_prediction_matrix, axis=1)[:, ::-1][:, 1:31]
cbcf_reclist = np.argsort(cbcf_prediction_matrix, axis=1)[:, ::-1][:, 1:31]
# print("cf", cf_reclist)
# print("cb", cb_reclist)
# print("cbcf", cbcf_reclist)

content_name_csv = []
content_name_miss = np.zeros(item_n)
for i in range(len(correct_value)):
    content_name_csv.append(content_name[int(test_data[i][1])])
    if cb_prediction[i] != correct_value[i]:
        content_name_miss[int(test_data[i][1])] += 1

with open('movie_len_cb.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(content_name_csv)
    writer.writerow(cb_prediction)
    writer.writerow(cf_prediction)
    writer.writerow(cbcf_prediction)
    writer.writerow(correct_value)

test_data_roc = deepcopy(correct_value)
for i in range(len(correct_value)):
    if correct_value[i] > 3:
        test_data_roc[i] = 1
    else:
        test_data_roc[i] = 0

cf_roc = deepcopy(cf_prediction)
for i in range(len(cf_prediction)):
    if cf_roc[i] < 3:
        cf_roc[i] = 0
    elif cf_roc[i] > 4:
        cf_roc[i] = 1
    else:
        cf_roc[i] -= 3

cbcf_roc = deepcopy(cbcf_prediction)
for i in range(len(cbcf_prediction)):
    if cbcf_roc[i] < 3:
        cbcf_roc[i] = 0
    elif cbcf_roc[i] > 4:
        cbcf_roc[i] = 1
    else:
        cbcf_roc[i] -= 3

# FPR, TPR(, しきい値) を算出
print(test_data_roc)
print(cf_roc)
fpr, tpr, thresholds = metrics.roc_curve(test_data_roc, cf_roc)

# ついでにAUCも
auc = metrics.auc(fpr, tpr)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='CF (AUC = %.2f)'%auc)
plt.legend()
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# FPR, TPR(, しきい値) を算出
print(cbcf_roc)
print(test_data_roc)
fpr, tpr, thresholds = metrics.roc_curve(test_data_roc, cbcf_roc)

# ついでにAUCも
auc = metrics.auc(fpr, tpr)
print(auc)

# ROC曲線をプロット
plt.plot(fpr, tpr, label='CBCF (AUC = %.2f)'%auc)
plt.legend()
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.savefig('movie_cbcf_roc_curve.png')
plt.show()