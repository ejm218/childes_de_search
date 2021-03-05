# IMPORT CHILDES SEARCH FUNCTIONS
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

# IMPORT THE CORPORA FOR USE IN THE SEARCH
# INCLUSION CRITERIA: AGES 1-6
chang1 = CHILDESCorpusReader(corpus_root, "Chang1/.*.xml") # youngest age 3;0
chang2 = CHILDESCorpusReader(corpus_root, "Chang2/.*.xml") # youngest age 3;0
changplay = CHILDESCorpusReader(corpus_root, "ChangPlay/.*.xml") # youngest age 3;0
erbaugh = CHILDESCorpusReader(corpus_root, "Erbaugh/.*.xml") # youngest age 2;0
lizhou = CHILDESCorpusReader(corpus_root, "LiZhou/.*.xml") # youngest age 3;0
tong = CHILDESCorpusReader(corpus_root, "Tong/.*.xml") # youngest age 1;7
zhou1 = CHILDESCorpusReader(corpus_root, "Zhou1/.*.xml") # youngest age 3;0
zhou2 = CHILDESCorpusReader(corpus_root, "Zhou2/.*.xml") # youngest age 3;0
zhou3 = CHILDESCorpusReader(corpus_root, "Zhou3/.*.xml") # youngest age 0;8
corpora = [chang1, chang2, changplay, erbaugh, lizhou, tong, zhou1, zhou2, zhou3]

# what do I want to find out
# when de emerges
# what types of items it is used to modify
np_search = "n:?[a-z]*|pro:?[a-z]*" # finds all nouns and pronouns
vp_search = "v:?[a-z]*" # finds all verbs

#data = pd.DataFrame(columns = ["filename", "age", "full_utterance", "search_term_tag", "preceding_item", "preceding_item_type", "succeeding_item", "succeeding_item_type"])
#for corpus in corpora:
#    corpus_data = corpus_search_MOR(corpus, "的")
#    for item in corpus_data:
#        data = data.append(corpus_data, ignore_index=True)
data = create_df(multicorpus_search_MOR(corpora, "的(?!(?:时候|话))"))
data = data.replace({"preceding_item_type": np_search}, "NP", regex=True)
data = data.replace({"succeeding_item_type": np_search}, "NP", regex=True)
data = data.replace({"preceding_item_type": vp_search}, "VP", regex=True)
data = data.replace({"succeeding_item_type": vp_search}, "VP", regex=True)
print(data.head())
print(data.tail())

# INTERIM SUMMARY OF DataFrame
print(f"DATA SUMMARY \n Age range: {data['age'].min()}-{data['age'].max()} months\n Number of head types: {data['succeeding_item_type'].nunique()}\n There are {len(data)} rows in the dataframe.\n Now printing first five rows: ")
#modifier_types = data["preceding_item_type"].value_counts()
#print(type(modifier_types))
#head_types = data["succeeding_item_type"].value_counts()
#print(head_types)
#variety = data["preceding_item_type"].nunique()
#print(variety)
print(data.head())

ages = list(set(data["age"].tolist()))
count_data = {
    "age": [],
    "total_de": [],
    "np_head": [],
    "vp_head": [],
    "sentence_final": [],
    "prec_adj": [],
    "prec_np": [],
    "prec_vp": []
    }
#print(data.groupby("age").succeeding_item_type.nunique())
for age in ages:
    target_data = data[(data["age"] == age)]
    count_data["age"].append(age)
    count_data["total_de"].append(len(target_data))
    count_data["np_head"].append(len(target_data[(target_data["succeeding_item_type"] == "NP")]))
    count_data["vp_head"].append(len(target_data[(target_data["succeeding_item_type"] == "VP")]))
    count_data["sentence_final"].append(len(target_data[(target_data["succeeding_item_type"] == "NA")]))
    count_data["prec_adj"].append(len(target_data[(target_data["preceding_item_type"] == "adj")]))
    count_data["prec_np"].append(len(target_data[(target_data["preceding_item_type"] == "NP")]))
    count_data["prec_vp"].append(len(target_data[(target_data["preceding_item_type"] == "VP")]))
print(count_data)
count_data = create_df(count_data)
count_data = count_data.sort_values(by="age")

#CREATING PLOT(s)
count_data.plot("age", ["sentence_final", "vp_head", "np_head"], kind="bar")
plt.savefig("headtype_raw.png")
count_data.plot("age", "total_de", kind="scatter")
plt.savefig("total_de_raw.png")

# to do:
# create new DF where heads are represented as percentages of total de counts
count_data_normed = count_data.copy()
for i in range(len(count_data_normed)):
    count_data_normed.iloc[i]
