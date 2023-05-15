import csv
import numpy as np
import math

data = []
N = 100
cols = 50

def read_csv():
    with open('./IRSys23_Docs.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

data = read_csv()
# print(type(data)) # <class 'list'>

def make_fr(data):
    fr_data = []
    for row in data:
        while len(row) < cols:
            row.append(0)
        fr_row = [0] * cols
        for i in range(cols):
            count = 0
            for j in range(cols):
                if int(row[j]) == i+1:
                    count += 1
            fr_row[i] = count
        fr_data.append(fr_row)
    return fr_data


def make_tf(fr_data):
    tf = []
    for row in fr_data:
        max_idx = np.argmax(row)
        tf_row = [0] * cols
        for c in range(cols):
            tf_row[c] = format(float(row[c])/float(row[max_idx]),'.8f')
        tf.append(tf_row)
    return tf


def make_idf(data):
    idf = []
    for c in range(cols):
        count = 0
        for r in range(N):
            if data[r][c] != 0:
                count +=1
        idf.append(count)
    idf = [format(math.log10(N/i),'.8f') for i in idf]
    return idf

def make_tf_idf(tf,idf):
    tf_idf = []
    for row in tf:
        tf_idf_row = [0] * cols
        for c in range(cols):
            tf_idf_row[c] = format(float(row[c]) * float(idf[c]),'.4f') 
        tf_idf.append(tf_idf_row)
    return tf_idf

def write_csv(tf_idf):
    with open('D213158.csv','w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(tf_idf)
    print("CSV DocTermMatrix file '{}' has been created.".format('D213158.csv') )

fr_data = make_fr(data)
tf_data = make_tf(fr_data)
idf_data = make_idf(fr_data)
tf_idf_data = make_tf_idf(tf_data,idf_data)


# test = int(input("put in test case: "))
# print(f"\n{fr_data[test]}") # [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
# print(f"\n{tf_data[test]}") # ['0.0000', '0.0000', '0.0588', '0.0000', '0.0000', '0.0000', '0.0588', '0.0000', '0.0000', '0.0000', '0.0588', '0.0000', '0.0588', '0.0588', '0.0000', '0.0000', '0.0000', '0.1176', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0588', '0.0000', '0.0588', '0.0000', '0.0588', '0.0588', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0588', '0.0000', '0.0588', '0.0588', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000']
# print(f"\n{idf_data}") # ['0.4949', '0.5686', '0.8539', '0.7212', '0.5686', '1.3010', '0.4202', '0.7696', '0.7212', '0.6021', '0.4318', '0.6778', '1.2218', '0.4318', '0.6576', '1.3979', '0.7959', '1.0969', '0.4559', '0.5229', '0.9586', '0.6778', '0.8539', '0.6990', '1.3979', '0.7212', '0.4437', '1.0969', '1.0458', '0.6198', '0.4089', '0.8239', '0.9586', '1.0969', '0.5850', '0.5229', '0.4202', '0.4318', '1.2218', '0.7212', '0.4685', '0.4437', '0.4815', '1.0969', '0.6383', '0.5086', '1.0000', '0.5528', '0.5229', '0.5850']
# print(f"\n{tf_idf_data[test]}") # ['0.0000', '0.0000', '0.0502', '0.0000', '0.0000', '0.0000', '0.0247', '0.0000', '0.0000', '0.0000', '0.0254', '0.0000', '0.0718', '0.0254', '0.0000', '0.0000', '0.0000', '0.1290', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0615', '0.0000', '0.0240', '0.0000', '0.0564', '0.0645', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0275', '0.0000', '0.0283', '0.0645', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000']
write_csv(tf_idf_data)