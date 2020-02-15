import numpy as np
import sys
import sklearn.preprocessing
import preprocessing
import csv

with open("mappings.csv", 'r') as file:
    read = csv.reader(file)
    data = list(read)
    np_arr = np.asarray(data)

# Delete column headers
mappings_arr = np.delete(np_arr, 0, axis=0)

icd9 = mappings_arr[:, 0]
icd10 = mappings_arr[:, 1]

# Process only the Dx Code 1 column
# I don't process other 2 columns as there are too many NULL values, esp col 3
temp_diag_code1 = preprocessing.new_arr[:, 26]
diag_code = np.array([], dtype = str)
temp_diag_code = np.array([], dtype = str)


# Deal with icd9 values that are not in the mappings
# What a pain in the ass -_-
for i in temp_diag_code1:
    if i == '714':
        i = np.array('714.4')

    if i == '715.97':
        i = np.array('715.95')

    if i == '715.98':
        i = np.array('715.96')

    if i == '696':
        i = np.array('696.2')

    if i == '716.95':
        i = np.array('714.4')

    if i == '716.96':
        i = np.array('714.4')

    if i == '719.96':
        i = np.array('714.4')

    if i == '719.95':
        i = np.array('715.95')

    if i == '733.4':
        i = np.array('715.95')

    if i == '754.3':
        i = np.array('715.95')

    if i == '808':
        i = np.array('820.21')

    if i == '820':
        i = np.array('820.21')

    if i == '823.2':
        i = np.array('820.21')

    if i == '996.4':
        i = np.array('996.44')

    temp_diag_code = np.append(temp_diag_code, [i], axis = 0)


# Convert all icd codes to ICD10 format
for i in temp_diag_code:
    check = i in icd9
    count = -1
    if check == True: # then convert icd9 to icd10
        list_index = np.where(icd9 == i)
        list_index2 = np.asarray(list_index)
        list_index3 = list_index2[0]
        list_index4 = icd10[list_index3]
        code = list_index4[0]
    else:
        code = i
    diag_code = np.append(diag_code, [code], axis = 0)

# print(diag_code.shape) #=> (4310, )


# One hot encoding the icd10 grouped codes
#uniqueValues, numbers = np.unique(diag_code, return_counts = True)
#for i in range(len(numbers)):
#    if numbers[i] > 200:
#        print("{}   {}      {}".format(i, uniqueValues[i], numbers[i]))


# Group ICD 10 codes
# Put all ICD 10 codes into 5 categories, 4 of them based on count > 200 or not, and the rest is other
icd10_one = np.array([], dtype = int)
icd10_two = np.array([], dtype = int)
icd10_three = np.array([], dtype = int)
icd10_four = np.array([], dtype = int)
icd10_other = np.array([], dtype = int)

one = [1]
zero = [0]
for i in diag_code:
    if i == 'M16.9':
        icd10_one = np.append(icd10_one, one, axis = 0)
        icd10_two = np.append(icd10_two, zero, axis = 0)
        icd10_three = np.append(icd10_three, zero, axis = 0)
        icd10_four = np.append(icd10_four, zero, axis = 0)
        icd10_other = np.append(icd10_other, zero, axis = 0)
    elif i == 'M17.9':
        icd10_one = np.append(icd10_one, zero, axis = 0)
        icd10_two = np.append(icd10_two, one, axis = 0)
        icd10_three = np.append(icd10_three, zero, axis = 0)
        icd10_four = np.append(icd10_four, zero, axis = 0)
        icd10_other = np.append(icd10_other, zero, axis = 0)
    elif i == 'M25.551':
        icd10_one = np.append(icd10_one, zero, axis = 0)
        icd10_two = np.append(icd10_two, zero, axis = 0)
        icd10_three = np.append(icd10_three, one, axis = 0)
        icd10_four = np.append(icd10_four, zero, axis = 0)
        icd10_other = np.append(icd10_other, zero, axis = 0)
    elif i == 'M25.561':
        icd10_one = np.append(icd10_one, zero, axis = 0)
        icd10_two = np.append(icd10_two, zero, axis = 0)
        icd10_three = np.append(icd10_three, zero, axis = 0)
        icd10_four = np.append(icd10_four, one, axis = 0)
        icd10_other = np.append(icd10_other, zero, axis = 0)
    else: # other
        icd10_one = np.append(icd10_one, zero, axis = 0)
        icd10_two = np.append(icd10_two, zero, axis = 0)
        icd10_three = np.append(icd10_three, zero, axis = 0)
        icd10_four = np.append(icd10_four, zero, axis = 0)
        icd10_other = np.append(icd10_other, one, axis = 0)


#icd10_one = icd10_one.reshape(icd10_one.shape[0], 1)
#icd10_two = icd10_two.reshape(icd10_two.shape[0], 1)
#icd10_three = icd10_three.reshape(icd10_three.shape[0], 1)
#icd10_four = icd10_four.reshape(icd10_four.shape[0], 1)
#icd10_other = icd10_other.reshape(icd10_other.shape[0], 1)

#print(icd10_one.shape)
#print(icd10_two.shape)
#print(icd10_three.shape)
#print(icd10_four.shape)
#print(icd10_other.shape)
