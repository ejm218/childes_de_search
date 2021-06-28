#THIS CODE IMPORTS THE PREVIOUSLY-CREATED DATA SET FROM A CSV FILE AND CREATES FURTHER SUBSETS TO ANALYSE
import os
import matplotlib as plt
import nltk
import re
from nltk.corpus.reader import CHILDESCorpusReader
from collections import Counter
import pandas as pd
os.chdir("/Users/academic/Documents/childes_de_search")
functions_file = open("childes_search.py", "r")
functions_file = functions_file.read()
exec(functions_file)
data = pd.read_csv("full_data.csv")

# SUBSETS
tong_data = data[data.filename.str.contains("Tong.*")==True]
print(f"There are {len(tong_data)} de utterances in the Tong dataset.")
erbaugh_data = data[data.filename.str.contains("Erbaugh.*")==True]
print(f"There are {len(erbaugh_data)} de utterances in the Erbaugh dataset.")
zhou1_data = data[data.filename.str.contains("Zhou1.*")==True]
print(f"There are {len(zhou1_data)} de utterances in the Zhou 1 dataset.")
zhou3_data = data[data.filename.str.contains("Zhou3.*")==True]
print(f"There are {len(zhou3_data)} de utterances in the Zhou 3 dataset.")
data_24to30 = pd.concat([tong_data, erbaugh_data, zhou3_data])
data_24to30 = data_24to30[data_24to30.age >= 24]
data_24to30 = data_24to30[data_24to30.age <= 30]
print(f"There are {len(data_24to30)} de utterances in the 24-30 month dataset.")
data_36to48 = data[data.age >= 36]
data_36to48 = data_36to48[data_36to48.age <= 48]
print(f"There are {len(data_36to48)} de utterances in the 36-48 month dataset.")

# CREATE FREQUENCY DATASETS
tong_head_count_data = pd.crosstab(index=tong_data["age"], columns=tong_data["succeeding_item_type"], normalize="index").reset_index()
tong_spec_count_data = pd.crosstab(index=tong_data["age"], columns=tong_data["preceding_item_type"], normalize="index").reset_index()
print(f"Created DF tong_head_count_data with {len(tong_head_count_data)} rows.")
print(f"Created DF tong_spec_count_data with {len(tong_spec_count_data)} rows.")
erbaugh_head_count_data = pd.crosstab(index=erbaugh_data["age"], columns=erbaugh_data["succeeding_item_type"], normalize="index").reset_index()
erbaugh_spec_count_data = pd.crosstab(index=erbaugh_data["age"], columns=erbaugh_data["preceding_item_type"], normalize="index").reset_index()
print(f"Created DF erbaugh_head_count_data with {len(erbaugh_head_count_data)} rows.")
print(f"Created DF erbaugh_spec_count_data with {len(erbaugh_spec_count_data)} rows.")
zhou3_head_count_data = pd.crosstab(index=zhou3_data["age"], columns=zhou3_data["succeeding_item_type"], normalize="index").reset_index()
zhou3_spec_count_data = pd.crosstab(index=zhou3_data["age"], columns=zhou3_data["preceding_item_type"], normalize="index").reset_index()
print(f"Created DF zhou3_head_count_data with {len(zhou3_head_count_data)} rows.")
print(f"Created DF zhou3_spec_count_data with {len(zhou3_spec_count_data)} rows.")
zhou1_head_count_data = pd.crosstab(index=zhou1_data["age"], columns=zhou1_data["succeeding_item_type"], normalize="index").reset_index()
zhou1_spec_count_data = pd.crosstab(index=zhou1_data["age"], columns=zhou1_data["preceding_item_type"], normalize="index").reset_index()
print(f"Created DF zhou1_head_count_data with {len(zhou1_head_count_data)} rows.")
print(f"Created DF zhou1_head_count_data with {len(zhou1_head_count_data)} rows.")
etz_head_count_data = pd.crosstab(index=data_24to30["age"], columns=data_24to30["succeeding_item_type"], normalize="index").reset_index()
etz_spec_count_data = pd.crosstab(index=data_24to30["age"], columns=data_24to30["preceding_item_type"], normalize="index").reset_index()
print(f"Created DF etz_head_count_data with {len(etz_head_count_data)} rows. This is a combination of the Erbaugh, Tong, and Zhou3 dataframes, comprising data from {etz_head_count_data.age.min()} to {etz_head_count_data.age.max()} months.")
print(f"Created DF etz_spec_count_data with {len(etz_spec_count_data)} rows. This is a combination of the Erbaugh, Tong, and Zhou3 dataframes, comprising data from {etz_spec_count_data.age.min()} to {etz_spec_count_data.age.max()} months.")
older_head_count_data = pd.crosstab(index=data_36to48["age"], columns=data_36to48["succeeding_item_type"], normalize="index").reset_index()
older_spec_count_data = pd.crosstab(index=data_36to48["age"], columns=data_36to48["preceding_item_type"], normalize="index").reset_index()
print(f"Created DF older_head_count_data with {len(older_head_count_data)} rows. This is a combination of all relevant corpora containing data from 36-48 months.")
print(f"Created DF older_spec_count_data with {len(older_spec_count_data)} rows. This is a combination of all relevant corpora containing data from 36-48 months.")
all_spec_count_data = pd.crosstab(index=data["age"], columns=data["preceding_item_type"], normalize="index").reset_index()

# CALCULATE REGRESSION LINE AKA LINE OF BEST FIT
from statistics import mean
def get_regression_line(x ,y ):
    m = (((mean(x)*mean(y)) - mean(x*y)) /
         ((mean(x)*mean(x)) - mean(x*x)))
    b = mean(y) - m*mean(x)
    regression_line = [(m*value)+b for value in x]
    return regression_line

#CREATING PLOT(s)
import matplotlib.pyplot as plt
print("Now generating plots...")
# development of items preceding de
# in the Tong corpus
tong_spec_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Proportion of Spec types in 'de' utterances")
plt.title("Items preceding 'de' in the Tong corpus")
plt.savefig("tong_preceding.png")
# in the Erbaugh corpus
erbaugh_spec_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Proportion of Spec types in 'de' utterances")
plt.title("Items preceding 'de' in the Erbaugh corpus")
plt.savefig("erbaugh_preceding.png")
# in the Zhou 1 corpus
zhou1_spec_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Proportion of Spec types in 'de' utterances")
plt.title("Items preceding 'de' in the Zhou 1 corpus")
plt.savefig("zhou1_preceding.png")
# in the Zhou 3 corpus
zhou3_spec_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Proportion of Spec types in 'de' utterances")
plt.title("Items preceding 'de' in the Zhou 3 corpus")
plt.savefig("zhou3_preceding.png")
# in the ETZ combined corpora
etz_spec_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Proportion of Spec types in 'de' utterances")
plt.title("Items preceding 'de' in the Erbaugh, Tong, Zhou 3 corpora (combined)")
plt.savefig("etz_preceding.png")
# development of items succeeding de i.e. head types
# in the Tong corpus
tong_head_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Types of items following 'de'")
plt.title("Items following 'de' in the Tong corpus")
plt.savefig("tong_heads.png")
# in the Erbaugh corpus
erbaugh_head_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Types of items following 'de'")
plt.title("Items following 'de' in the Erbaugh corpus")
plt.savefig("erbaugh_heads.png")
# in the Zhou 1 corpus
zhou1_head_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Types of items following 'de'")
plt.title("Items following 'de' in the Zhou 1 corpus")
plt.savefig("zhou1_heads.png")
# in the Zhou 3 corpus
zhou3_head_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Types of items following 'de'")
plt.title("Items following 'de' in the Zhou 3 corpus")
plt.savefig("zhou3_heads.png")
# in the ETZ combined corpora
etz_head_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Types of items following 'de'")
plt.title("Items following 'de' in the Erbaugh, Tong, Zhou 3 corpora (combined)")
plt.savefig("etz_heads.png")

# Erbaugh total number of entries per age 
erbaugh_total_monthly = dict()
for i in set(erbaugh_data["age"].values):
	erbaugh_total_monthly[i] = len(erbaugh_data.loc[erbaugh_data["age"] == i])

# Add a column to the Tong DF that contains total number of utterances for the file
for i in tong_data["age"].values:
    fileid = tong_data.loc[tong_data["age"] == i]["filename"].values[0]
    proportion_de = int(tong_data["filename"].value_counts()[fileid]) / len(tong.sents(fileids=fileid, speaker="CHI"))
    for i in tong_data:
        tong_data["de_proportion"][i] = proportion_de
