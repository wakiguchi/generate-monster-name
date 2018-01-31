import numpy as np
import random
import tensorflow as tf
import pandas as pd
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
import os
from train_name import DNN
from train_name import zscore
from train_name import dataprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    '''
    Data processing!
    '''
    # Set seed
    tf.set_random_seed(0)

    # Set random number
    rand_low = 10
    rand_high = 110
    rand_1 = 2
    rand_2 = 7
    rand_3 = 8
    p_H = random.randint(rand_low, rand_high) + random.randint(0, rand_1)*random.randint(0, rand_2)*random.randint(0, rand_3)
    p_A = random.randint(rand_low, rand_high) + random.randint(0, rand_1)*random.randint(0, rand_2)*random.randint(0, rand_3)
    p_B = random.randint(rand_low, rand_high) + random.randint(0, rand_1)*random.randint(0, rand_2)*random.randint(0, rand_3)
    p_C = random.randint(rand_low, rand_high) + random.randint(0, rand_1)*random.randint(0, rand_2)*random.randint(0, rand_3)
    p_D = random.randint(rand_low, rand_high) + random.randint(0, rand_1)*random.randint(0, rand_2)*random.randint(0, rand_3)
    p_S = random.randint(rand_low, rand_high) + random.randint(0, rand_1)*random.randint(0, rand_2)*random.randint(0, rand_3)
    type_list1=["くさ","どく","みず","ひこう","ほのお","むし","こおり","エスパー","かくとう","ドラゴン","あく","ゴースト","はがね","フェアリー","じめん","いわ","ノーマル","でんき"]
    type_list2=type_list1
    type_list2.append("無")
    type1_num = random.randint(0, len(type_list1)-2)
    type2_num = random.randint(0, len(type_list1)-1)
    if(type1_num == type2_num):
        type2_num = len(type_list1)-1

    # Load data set
    df = pd.read_csv('data/database_pokemon_mini.csv')
    # Set random data
    df_gene = pd.DataFrame([[len(df['i']),801,"!",type_list1[type1_num],type_list2[type2_num],p_A,p_B,p_C,p_D,p_H,p_S]], columns=["","i","name","type1","type2","v_a","v_b","v_c","v_d","v_h","v_s"])
    # Merge data
    df = pd.concat([df, df_gene])
    name_list,x_type1,x_type2,x_bs,Y_all,char_name_all,names = dataprocessing(df)

    count = 0 #count is number of letters
    key = 0   #key matches characters and vectors
    result = ""
    data_return=""
    while(result != "!"):
        '''
        Regression to predict names
        '''
        predict_count = count
        predict_key = key
        XX = []
        XX.extend(x_bs[len(x_bs)-1])
        XX.extend(x_type1[len(x_bs)-1])
        XX.extend(x_type2[len(x_bs)-1])
        j = 0
        chara = names[predict_key]
        chara = chara.replace("0=", "")

        for char_tmp in char_name_all:
            if char_tmp == chara:
                predict_key = j
            j += 1
        XX.extend(Y_all[predict_key])
        XX.append(predict_count)
        X_all = []
        X_all.append(XX)

        '''
        Setting model!
        '''
        model = DNN(n_in=len(X_all[0]),n_hiddens=[400, 400, 200],n_out=len(Y_all[0]))
        '''
        Prediction!
        '''
        model.forward(X_all[0:1],p_keep=1.0)

        '''
        Show Result
        '''
        key_list = list(model._history['argmax'])
        result_name_all = []
        for each_name in key_list:
            key = each_name
            result_name_all.append(names[key])
        result = ''
        for x in result_name_all:
            result += x
        result = result.replace("0=", "")
        data_return = data_return + result
        count += 1

    #Set rank
    p_sum = p_H + p_A + p_B + p_C + p_D + p_S
    if(p_sum >= 700):
        rank = "S"
    elif(p_sum >= 650):
        rank = "A+"
    elif(p_sum >= 600):
        rank = "A"
    elif(p_sum >= 550):
        rank = "B+"
    elif(p_sum >= 500):
        rank = "B"
    elif(p_sum >= 450):
        rank = "C+"
    elif(p_sum >= 400):
        rank = "C"
    elif(p_sum >= 300):
        rank = "D"
    else:
        rank = "E"

    data_return = (data_return + "\nタイプ1:" + type_list1[type1_num] + "\nタイプ2:" + type_list2[type2_num]
                     + "\nHP:" + str(p_H) + "\n攻撃:" + str(p_A) + "\n防御:" + str(p_B) + "\n特攻:" + str(p_C)
                     + "\n特防:" + str(p_D) + "\n素早:" + str(p_S) + "\n合計:" + str(p_sum) + "(" + rank + "ランク!)")

    print(data_return)
