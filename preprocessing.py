import numpy as np
import sys
import sklearn.preprocessing

import csv
with open("snf.csv", 'r') as file:
    read = csv.reader(file)
    data = list(read)
    np_arr = np.asarray(data)

# Delete columns headers
# params: objects, row, col. axis = 0 means deleting first row, 1 is first col
no_header_arr = np.delete(np_arr, 0, axis=0)

# Process State (the foremost because I delete all entries that is not of New York state)
state_indices = np.where(no_header_arr[:, 3] != "NY")
# Make a new matrix without entries from other than New York
temp_arr = np.delete(no_header_arr, state_indices, axis = 0)


# Delete rows where height = null
null_height = np.where(temp_arr[:, 32] == "NULL")
new_arr = np.delete(temp_arr, null_height, axis = 0)

# print("Shape of the new array is: ") => (4310,35)
# print(new_arr.shape)


# Process column by column

# Process Gender
# Encode M = 0, F=1
gender = np.array([], dtype = int)
for i in new_arr[:, 2]:
    if i == "M":
        i = 0
    else:
        i = 1
    gender = np.append(gender, [i], axis = 0)

# gender = gender.reshape(gender.shape[0], 1)
# print(gender.shape) => (4315,)

# Process Length of Stay
# Normalizing all values so that the model does not give too much bias on long length of stay
temp_length_of_stay = np.array([], dtype = float)
length_of_stay = np.array([], dtype = float)
for i in new_arr[:, 4]:
    i = float(i)
    temp_length_of_stay = np.append(temp_length_of_stay, [i], axis = 0)

for j in temp_length_of_stay:
    j = j/50
    length_of_stay = np.append(length_of_stay, [j], axis = 0)

#length_of_stay = length_of_stay.reshape(gender.shape[0], 1)
# print(length_of_stay.shape) => (4315,)
# print(length_of_stay)


# Process Pat_EthGrp
# As there are close to 4300 entries featuring Not Hispanic and Latino, Not H&L = 0, all other = 1
ethnic = np.array([], dtype = int)
for i in new_arr[:, 5]:
    if i == "Not Hispanic or Latino":
        i = 0
    else:
        i = 1
    ethnic = np.append(ethnic, [i], axis = 0)

#ethnic = ethnic.reshape(ethnic.shape[0], 1)

# roughly 83 are Hispanic or Latino and Other
# count = 0
# for i in ethnic:
#    if  i == 1:
#        count +=1
# print(count)
# print(ethnic)
# print(ethnic.shape) => (4315,)


# Process Pat_Race
# One-hot encoding
white = np.array([], dtype = int)
black = np.array([], dtype = int)
other_race = np.array([], dtype = int)
a = [0]
for i in new_arr[:, 6]:
    if i == "White or Caucasian":
        i = 1
        white = np.append(white, [i], axis = 0)
        black = np.append(black, a, axis = 0)
        other_race = np.append(other_race, a, axis = 0)
    elif i == "Black or African American":
        i = 1
        white = np.append(white, a, axis = 0)
        black = np.append(black, [i], axis = 0)
        other_race = np.append(other_race, a, axis = 0)
    else:
        i = 1
        white = np.append(white, a, axis = 0)
        black = np.append(black, a, axis = 0)
        other_race = np.append(other_race, [i], axis = 0)

#white = white.reshape(white.shape[0], 1)
#black = black.reshape(black.shape[0], 1)
#other = other.reshape(other.shape[0], 1)

#print(other.shape)

# print(other[19]) => print 1 for other
# print(other.shape)


# All columns with binary values in the middle
lives_alone = new_arr[:,7]
asthma = new_arr[:,8]
afib = new_arr[:,9]
cad = new_arr[:,10]
chf = new_arr[:,11]
copd = new_arr[:,12]
diabetes = new_arr[:,13]
diabetes_by_dashbd = new_arr[:,14]
htn = new_arr[:,15]
obesity = new_arr[:,16]
ckd = new_arr[:,17]
cld = new_arr[:,18]
dprsn = new_arr[:,19]
ostpor = new_arr[:,20]
cl = new_arr[:,21]
lipid = new_arr[:,22]


# Process INDEX_DISCH_DISP_NM
# Column containing binary values: SNF = 1, otherwise 0
snf = np.array([], dtype = int)
for i in new_arr[:, 23]:
    if "SNF" in i:
        i = 1
    else:
        i = 0
    snf = np.append(snf, [i], axis = 0)

#snf = snf.reshape(snf.shape[0], 1)

# print(snf)
# print(snf.shape) => (4310,)

# Create 3 new variables from the column ProcName1
temp = new_arr[:, 24]
proc_name = np.array([], dtype = int)
# Convert all string to lowercase
for i in temp:
    i = i.lower()
    proc_name = np.append(proc_name, [i], axis = 0)


# Variable 1: which side the surgery was performed on (left, right, both or unknown)
# if left => 1, right => 1, both => left = 1 and right = 1, unknown => left = 0 and right = 0
left = np.array([], dtype = int)
right = np.array([], dtype = int)
both = np.array([], dtype = int)
unknown = np.array([], dtype = int)
for i in proc_name:
    if "left" in i:
        left_side = 1
        all = 0
        left = np.append(left, [left_side], axis = 0)
        right = np.append(right, [all], axis = 0)
        both = np.append(both, [all], axis = 0)
        unknown = np.append(unknown, [all], axis = 0)
    elif "right" in i:
        right_side = 1
        all = 0
        left = np.append(left, [all], axis = 0)
        right = np.append(right, [right_side], axis = 0)
        both = np.append(both, [all], axis = 0)
        unknown = np.append(unknown, [all], axis = 0)
    elif "biomet" in i:
        both_side = 1
        all = 0
        left = np.append(left, [all], axis = 0)
        right = np.append(right, [all], axis = 0)
        both = np.append(both, [both_side], axis = 0)
        unknown = np.append(unknown, [all], axis = 0)
    else:
        unknown_side = 1
        all = 0
        left = np.append(left, [all], axis = 0)
        right = np.append(right, [all], axis = 0)
        both = np.append(both, [all], axis = 0)
        unknown = np.append(unknown, [unknown_side], axis = 0)

#left = left.reshape(left.shape[0], 1)
#right = right.reshape(right.shape[0], 1)
#both = both.reshape(both.shape[0], 1)
#unknown = unknown.reshape(unknown.shape[0], 1)

# print(left.shape)
# print(right.shape)
# print(both.shape)
# print(unknown.shape)


# Variable 2: surgery on knee or hip
hip = np.array([], dtype = int)
knee = np.array([], dtype = int)
for i in proc_name:
    if "hip" in i:
        h = 1
        hip = np.append(hip, [h], axis = 0)
        k = 0
        knee = np.append(knee, [k], axis = 0)
    elif "knee" in i:
        h = 0
        hip = np.append(hip, [h], axis = 0)
        k = 1
        knee = np.append(knee, [k], axis = 0)

#hip = hip.reshape(hip.shape[0], 1)
#knee = knee.reshape(knee.shape[0], 1)

# print(hip)
# print(knee)

# Variable 3: whether anterior approach was used or not
anterior = np.array([], dtype = int)
for i in proc_name:
    if "anterior" in i:
        a = 1
        anterior = np.append(anterior, [a], axis = 0)
    else:
        a = 0
        anterior = np.append(anterior, [a], axis = 0)

#anterior = anterior.reshape(anterior.shape[0], 1)
# print(anterior.shape) => (4310,1)

# Process age
# Make 90+ age into 90
temp_age = np.array([], dtype = float)
temp_age1 = np.array([], dtype = float)
age = np.array([], dtype = float)

for i in new_arr[:, 25]:
    if "90+" in i:
        i = 90
        temp_age = np.append(temp_age, [i], axis = 0)
    else:
        temp_age = np.append(temp_age, [i], axis = 0)

for i in temp_age:
    i = float(i)
    temp_age1 = np.append(temp_age1, [i], axis = 0)


# Normalize by diving by the max age
for i in temp_age1:
    i = i/90
    age = np.append(age, [i], axis = 0)

#age = age.reshape(age.shape[0], 1)
# print(age.shape) => (4315,1)
# print(age)


# Process the ICD codes. Turn all ICD9 to ICD10
# Throw away 2 cols Dx Code 2 and Dx Code 3
# See the icd_preprocessing.py script for more details


# Process Provider
# For all providers whose count > 100, they have their own column, otherwise they fall in the category other
# Enumerating counts:
# uniqueValues, count = np.unique(new_arr[:, 29], return_counts = True)
# for i in range(len(uniqueValues)):
#    print("{}    and count    {}".format(uniqueValues[i], count[i]))

provider_1 = np.array([], dtype = int)
provider_2 = np.array([], dtype = int)
provider_3 = np.array([], dtype = int)
provider_4 = np.array([], dtype = int)
provider_5 = np.array([], dtype = int)
provider_6 = np.array([], dtype = int)
provider_7 = np.array([], dtype = int)
provider_other = np.array([], dtype = int)

zero = [0]
for i in new_arr[:, 29]:
    if "EDWARDS,SAM D" in i: # count = 302
        edwards = 1
        provider_1 = np.append(provider_1, [edwards], axis = 0)
        provider_2 = np.append(provider_2, zero, axis = 0)
        provider_3 = np.append(provider_3, zero, axis = 0)
        provider_4 = np.append(provider_4, zero, axis = 0)
        provider_5 = np.append(provider_5, zero, axis = 0)
        provider_6 = np.append(provider_6, zero, axis = 0)
        provider_7 = np.append(provider_7, zero, axis = 0)
        provider_other = np.append(provider_other, zero, axis = 0)
    elif "RACHEL Q" in i: # count = 674
        rachel = 1
        provider_1 = np.append(provider_1, zero, axis = 0)
        provider_2 = np.append(provider_2, [rachel], axis = 0)
        provider_3 = np.append(provider_3, zero, axis = 0)
        provider_4 = np.append(provider_4, zero, axis = 0)
        provider_5 = np.append(provider_5, zero, axis = 0)
        provider_6 = np.append(provider_6, zero, axis = 0)
        provider_7 = np.append(provider_7, zero, axis = 0)
        provider_other = np.append(provider_other, zero, axis = 0)
    elif "GREENBERG,ALLISON" in i: # count = 775
        greenberg = 1
        provider_1 = np.append(provider_1, zero, axis = 0)
        provider_2 = np.append(provider_2, zero, axis = 0)
        provider_3 = np.append(provider_3, [greenberg], axis = 0)
        provider_4 = np.append(provider_4, zero, axis = 0)
        provider_5 = np.append(provider_5, zero, axis = 0)
        provider_6 = np.append(provider_6, zero, axis = 0)
        provider_7 = np.append(provider_7, zero, axis = 0)
        provider_other = np.append(provider_other, zero, axis = 0)
    elif "MCMYERS,DANIEL" in i: # count = 1218
        mcmyers = 1
        provider_1 = np.append(provider_1, zero, axis = 0)
        provider_2 = np.append(provider_2, zero, axis = 0)
        provider_3 = np.append(provider_3, zero, axis = 0)
        provider_4 = np.append(provider_4, [mcmyers], axis = 0)
        provider_5 = np.append(provider_5, zero, axis = 0)
        provider_6 = np.append(provider_6, zero, axis = 0)
        provider_7 = np.append(provider_7, zero, axis = 0)
        provider_other = np.append(provider_other, zero, axis = 0)
    elif "SHAPIRO,STEPHANIE" in i: # count = 826
        shapiro = 1
        provider_1 = np.append(provider_1, zero, axis = 0)
        provider_2 = np.append(provider_2, zero, axis = 0)
        provider_3 = np.append(provider_3, zero, axis = 0)
        provider_4 = np.append(provider_4, zero, axis = 0)
        provider_5 = np.append(provider_5, [shapiro], axis = 0)
        provider_6 = np.append(provider_6, zero, axis = 0)
        provider_7 = np.append(provider_7, zero, axis = 0)
        provider_other = np.append(provider_other, zero, axis = 0)
    elif "STEVENSON,STEPHEN" in i: # count = 270
        stevenson = 1
        provider_1 = np.append(provider_1, zero, axis = 0)
        provider_2 = np.append(provider_2, zero, axis = 0)
        provider_3 = np.append(provider_3, zero, axis = 0)
        provider_4 = np.append(provider_4, zero, axis = 0)
        provider_5 = np.append(provider_5, zero, axis = 0)
        provider_6 = np.append(provider_6, [stevenson], axis = 0)
        provider_7 = np.append(provider_7, zero, axis = 0)
        provider_other = np.append(provider_other, zero, axis = 0)
    elif "WOLBY,ROY" in i: # count = 148
        wolby = 1
        provider_1 = np.append(provider_1, zero, axis = 0)
        provider_2 = np.append(provider_2, zero, axis = 0)
        provider_3 = np.append(provider_3, zero, axis = 0)
        provider_4 = np.append(provider_4, zero, axis = 0)
        provider_5 = np.append(provider_5, zero, axis = 0)
        provider_6 = np.append(provider_6, zero, axis = 0)
        provider_7 = np.append(provider_7, [wolby], axis = 0)
        provider_other = np.append(provider_other, zero, axis = 0)
    else:
        other = 1
        provider_1 = np.append(provider_1, zero, axis = 0)
        provider_2 = np.append(provider_2, zero, axis = 0)
        provider_3 = np.append(provider_3, zero, axis = 0)
        provider_4 = np.append(provider_4, zero, axis = 0)
        provider_5 = np.append(provider_5, zero, axis = 0)
        provider_6 = np.append(provider_6, zero, axis = 0)
        provider_7 = np.append(provider_7, zero, axis = 0)
        provider_other = np.append(provider_other, [other], axis = 0)

# Dimensions of all provider arrays are (4310,1)
#provider_1 = provider_1.reshape(provider_1.shape[0], 1)
#provider_2 = provider_2.reshape(provider_2.shape[0], 1)
#provider_3 = provider_3.reshape(provider_3.shape[0], 1)
#provider_4 = provider_4.reshape(provider_4.shape[0], 1)
#provider_5 = provider_5.reshape(provider_5.shape[0], 1)
#provider_6 = provider_6.reshape(provider_6.shape[0], 1)
#provider_7 = provider_7.reshape(provider_7.shape[0], 1)
#provider_other = provider_other.reshape(provider_other.shape[0], 1)

# Process Reg Fsc 1
# All reg fsc whose count < 200 falls into other category

# Enumerating counts:
#uniqueVals, number = np.unique(new_arr[:, 30], return_counts = True)
#for i in range(len(number)):
#    if number[i] > 200:
#        print("{}   {}      {}".format(i, uniqueVals[i], number[i]))

reg_fsc_1 = np.array([], dtype = int)
reg_fsc_2 = np.array([], dtype = int)
reg_fsc_3 = np.array([], dtype = int)
reg_fsc_4 = np.array([], dtype = int)
reg_fsc_5 = np.array([], dtype = int)
reg_fsc_other = np.array([], dtype = int)

encode = [0]
for i in new_arr[:, 30]:
    if i == "BLUE CHOICE":
        blue_choice = 1
        reg_fsc_1 = np.append(reg_fsc_1, [blue_choice], axis = 0)
        reg_fsc_2 = np.append(reg_fsc_2, encode, axis = 0)
        reg_fsc_3 = np.append(reg_fsc_3, encode, axis = 0)
        reg_fsc_4 = np.append(reg_fsc_4, encode, axis = 0)
        reg_fsc_5 = np.append(reg_fsc_5, encode, axis = 0)
        reg_fsc_other = np.append(reg_fsc_other, encode, axis = 0)
    elif i == "BLUE SHIELD":
        blue_shield = 1
        reg_fsc_1 = np.append(reg_fsc_1, encode, axis = 0)
        reg_fsc_2 = np.append(reg_fsc_2, [blue_shield], axis = 0)
        reg_fsc_3 = np.append(reg_fsc_3, encode, axis = 0)
        reg_fsc_4 = np.append(reg_fsc_4, encode, axis = 0)
        reg_fsc_5 = np.append(reg_fsc_5, encode, axis = 0)
        reg_fsc_other = np.append(reg_fsc_other, encode, axis = 0)
    elif i == "MEDICARE":
        medicare = 1
        reg_fsc_1 = np.append(reg_fsc_1, encode, axis = 0)
        reg_fsc_2 = np.append(reg_fsc_2, encode, axis = 0)
        reg_fsc_3 = np.append(reg_fsc_3, [medicare], axis = 0)
        reg_fsc_4 = np.append(reg_fsc_4, encode, axis = 0)
        reg_fsc_5 = np.append(reg_fsc_5, encode, axis = 0)
        reg_fsc_other = np.append(reg_fsc_other, encode, axis = 0)
    elif i == "MEDICARE BLUE CHOICE":
        mec_blue_choice = 1
        reg_fsc_1 = np.append(reg_fsc_1, encode, axis = 0)
        reg_fsc_2 = np.append(reg_fsc_2, encode, axis = 0)
        reg_fsc_3 = np.append(reg_fsc_3, encode, axis = 0)
        reg_fsc_4 = np.append(reg_fsc_4, [mec_blue_choice], axis = 0)
        reg_fsc_5 = np.append(reg_fsc_5, encode, axis = 0)
        reg_fsc_other = np.append(reg_fsc_other, encode, axis = 0)
    elif i == "MVP PREFERRED GOLD":
        mvp = 1
        reg_fsc_1 = np.append(reg_fsc_1, encode, axis = 0)
        reg_fsc_2 = np.append(reg_fsc_2, encode, axis = 0)
        reg_fsc_3 = np.append(reg_fsc_3, encode, axis = 0)
        reg_fsc_4 = np.append(reg_fsc_4, encode, axis = 0)
        reg_fsc_5 = np.append(reg_fsc_5, [mvp], axis = 0)
        reg_fsc_other = np.append(reg_fsc_other, encode, axis = 0)
    else:
        reg_others = 1
        reg_fsc_1 = np.append(reg_fsc_1, encode, axis = 0)
        reg_fsc_2 = np.append(reg_fsc_2, encode, axis = 0)
        reg_fsc_3 = np.append(reg_fsc_3, encode, axis = 0)
        reg_fsc_4 = np.append(reg_fsc_4, encode, axis = 0)
        reg_fsc_5 = np.append(reg_fsc_5, encode, axis = 0)
        reg_fsc_other = np.append(reg_fsc_other, [reg_others], axis = 0)

# Dimensions of all reg_fsc is (4310,1)
#reg_fsc_1 = reg_fsc_1.reshape(reg_fsc_1.shape[0], 1)
#reg_fsc_2 = reg_fsc_2.reshape(reg_fsc_2.shape[0], 1)
#reg_fsc_3 = reg_fsc_3.reshape(reg_fsc_3.shape[0], 1)
#reg_fsc_4 = reg_fsc_4.reshape(reg_fsc_4.shape[0], 1)
#reg_fsc_5 = reg_fsc_5.reshape(reg_fsc_5.shape[0], 1)
#reg_fsc_other = reg_fsc_other.reshape(reg_fsc_other.shape[0], 1)

# I will not take the column Reg Fsc 2 into consideration when doing analysis
# This col has too many NULL values (850), and most of the common Reg Fsc 1
# are also common Reg Fsc 2
# All in all, substanstial info in Reg Fsc 2 is all captured by Reg Fsc 1


# Process height
# Deal with all 'bad' entries of height
temp_height = np.array([], dtype = float)
height = np.array([], dtype = float)
for i in new_arr[:, 32]:
    if i == "[167.6[":
        i = "5'6"
    elif "[62" in i:
        i = "5'2"
    elif "5f5i" in i:
        i = "5'5"
    elif "Simultaneous" in i:
        i = "5'7"
    temp_height = np.append(temp_height, [i], axis = 0)

# Convert height into decimal values of feet
for i in temp_height:
    x, y = i.split("'")
    feet = float(x)
    inches = float(y.replace('"', "")) # delete the "
    height_in_decimal_form = feet + (inches*1/12)
    height = np.append(height, [height_in_decimal_form], axis = 0)

#height = height.reshape(height.shape[0], 1)
# print(height)
# print(height.shape) #=> (4310, 1)


# Process weight
temp_weight = np.array([], dtype = int)
weight = np.array([], dtype = int)

for i in new_arr[:, 33]:
    i = float(i)
    temp_weight = np.append(temp_weight, [i], axis = 0)

max = 0
for n in temp_weight:
    if n > max:
        max = n

for j in temp_weight:
    j = j/max
    weight = np.append(weight, [j], axis = 0)

#weight = weight.reshape(weight.shape[0], 1)

#print(weight)
#print(weight.shape) #=> (4310,)


prev_snf_admission = new_arr[:,34]
