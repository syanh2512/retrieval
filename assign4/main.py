import csv
import numpy as np

cols = 5

def read_csv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def make_adja_matrix(data):
    adja_matrix = []
    for row in data:
        while len(row) < cols:
            row.append(0)
        fr_row = [0]*cols
        for i in range(cols):
            count = 0
            for j in range(cols):
                if int(row[j]) == i+1:
                    count +=1
            fr_row[i] = count
        adja_matrix.append(fr_row)
    return adja_matrix

def make_trproba_matrix(adja_matrix):
    for i, row in enumerate(adja_matrix):
        count = 0
        for j in range(cols):
            if row[j] != 0:
                count +=1
        for j in range(cols):
            row[j] = round(float(row[j])/float(count),4)
            # row[j] = format(float(row[j])/float(count),'.4f')
        adja_matrix[i] = row
    trproba_matrix = [[row[i] for row in adja_matrix] for i in range(len(adja_matrix[0]))]
    return trproba_matrix

def write_csv(trproba_matrix,file_path):
    with open(file_path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(trproba_matrix)
    print("\nCSV Transition Probability Matrix file '{}' has been created.\n".format(file_path) )


web_links = read_csv('./IRSys23_WebLinks.csv')
# print(f'{web_links}\n')
adja_matrix = make_adja_matrix(web_links)
# print(f'{np.array(adja_matrix)}\n')
trproba_matrix = make_trproba_matrix(adja_matrix)
print(f'{np.array(trproba_matrix)}\n')
# write_csv(trproba_matrix,'./T213158.csv')

trproba_matrix = np.array(trproba_matrix)

eigenvalues, eigenvectors = np.linalg.eig(trproba_matrix)

print("eigenvalue:\n", np.abs(eigenvalues))
print("eigenvectors:\n", np.abs(eigenvectors))

max_eigenvalue_index = np.argmax(eigenvalues)

# Retrieve the eigenvector corresponding to the maximum eigenvalue
max_eigenvalue = eigenvalues[max_eigenvalue_index]
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

print("\nMaximum Eigenvalue:\n", np.abs(max_eigenvalue))
print("Corresponding Eigenvector:\n", np.abs(max_eigenvector))