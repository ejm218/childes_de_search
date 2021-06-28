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
lizhou = CHILDESCorpusReader(corpus_root, "LiZhou/.*.xml") # youngest age 2;0
erbaugh = CHILDESCorpusReader(corpus_root, "Erbaugh/.*.xml") # youngest age 2;0
tong = CHILDESCorpusReader(corpus_root, "Tong/.*.xml") # youngest age 1;7
zhou1 = CHILDESCorpusReader(corpus_root, "Zhou1/.*.xml") # youngest age 3;0
zhou2 = CHILDESCorpusReader(corpus_root, "Zhou2/.*.xml") # youngest age 3;0
zhou3 = CHILDESCorpusReader(corpus_root, "Zhou3/.*.xml") # youngest age 0;8
corpora = [erbaugh, lizhou, tong, zhou1, zhou2, zhou3]

np_search = "n:?[a-z]*|pro:?[a-z]*" # finds all nouns and pronouns
vp_search = "v:?[a-z]*" # finds all verbs

data = create_df(multicorpus_search_MOR(corpora, "的(?!(?:时候|话))"))
data = data[data.age <= 48]
data = data.replace({"preceding_item_type": np_search}, "NP", regex=True)
data = data.replace({"succeeding_item_type": np_search}, "NP", regex=True)
data = data.replace({"preceding_item_type": vp_search}, "VP", regex=True)
data = data.replace({"succeeding_item_type": vp_search}, "VP", regex=True)
data = data.loc[data["succeeding_item_type"] != ""]
data = data.loc[data["succeeding_item_type"] != "poss"]
data = data.loc[data["succeeding_item_type"] != "chi"]
data = data.loc[data["succeeding_item_type"] != "cl"]
data.loc[data["succeeding_item"] == "是", "succeeding_item_type"] = "cop"
data.loc[data["succeeding_item_type"] == "sfp", "succeeding_item_type"] = "NA"
data.loc[data["succeeding_item_type"] == "adVP", "succeeding_item_type"] = "adv"
data.loc[data["succeeding_item_type"] == "co", "succeeding_item_type"] = "det"
data.loc[data["succeeding_item_type"] == "co:iNP", "succeeding_item_type"] = "NA"
data.loc[data["succeeding_item_type"] == "coNP", "succeeding_item_type"] = "NA"
data.loc[data["succeeding_item_type"] == "asp", "succeeding_item_type"] = "NA"
data.loc[data["succeeding_item_type"] == "quaNP", "succeeding_item_type"] = "adj"
data.loc[data["preceding_item"] == "是", "preceding_item_type"] = "cop"
data.loc[data["preceding_item_type"] == "sfp", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "co", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "co:iNP", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "coNP", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "asp", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "quaNP", "preceding_item_type"] = "adj"
data.loc[data["preceding_item_type"] == "L2", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "poss", "preceding_item_type"] = "adj"
data.loc[data["preceding_item_type"] == "", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "cl", "preceding_item_type"] = "adj"
data.loc[data["preceding_item_type"] == "cleft", "preceding_item_type"] = "other"
data.loc[data["preceding_item_type"] == "oNP", "preceding_item_type"] = "other"

print(f"The full dataset contains {len(data)} utterances between {data.age.min()} and {data.age.max()} months.")
data.to_csv("full_data.csv")
print("Data has been saved to a CSV file.")
#modifier_types = data["preceding_item_type"].value_counts()
#print(type(modifier_types))
#head_types = data["succeeding_item_type"].value_counts()
#print(head_types)
#variety = data["preceding_item_type"].nunique()
#print(variety)
