import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

retrieve_data = pd.read_csv('./IRSys23_RetrievedDocs.csv',header=None)
relate_data = pd.read_csv('./IRSys23_RelatedDocs.csv',header=None)

# print(f'{retrieve_data.shape} & {relate_data.shape}')

element_found = 0
recall_element_total = 5
retrieve_element = 0
precision_recall = []
AP = 0.00


for el1 in retrieve_data.iloc[:,0]:
    retrieve_element +=1
    for el2 in relate_data.iloc[0,:]:
        if el1 == el2:
            element_found +=1
            # print(f'\n{element_found} element found, P_element = {retrieve_element}, R_element = {recall_element_total}')
            P = (element_found/retrieve_element)
            R = (element_found/recall_element_total)
            precision_recall.append([P,R])
            if retrieve_element <= recall_element_total:
                AP += P/recall_element_total
                R_precision = R

precision_recall.append([0.00,1.00])
print(f'\n[[P, R]] = \n{precision_recall}')

re = 0.00
interpolation = []
p_sum = 0
for p,r in precision_recall:
    while re <= r:
        p_sum += p
        interpolation.append([p,re])
        re=(re*100+10)/100

    
mp = round(p_sum/11,2)
print(f'\nRecall - Precision with Interpolation:\n{interpolation}')
print(f'\nAverage Precision = {AP}, 11 point AP = {mp}, R-Precision = {R_precision}')



interpolation = pd.DataFrame(interpolation)
interpolation.to_csv('p_r_graph.csv', header=None, index=None)
P_data = interpolation.iloc[:,0]
R_data = interpolation.iloc[:,1]
sns.lineplot(x=R_data*100, y=P_data*100, marker='o',color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(0,120)
plt.ylim(0,120)
plt.savefig('p_r_graph.png',format='png')
# plt.show()

