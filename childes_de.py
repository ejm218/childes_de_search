# IMPORT CHILDES SEARCH FUNCTIONS
import os
import matplotlib as plt
import nltk
import re
from nltk.corpus.reader import CHILDESCorpusReader
from collections import Counter
import pandas as pd
os.chdir("/Users/academic/Documents/childes_de_search")
functions_file = open("childes_search_functions.py", "r")
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

data = create_df(multicorpus_search_MOR(corpora, "的(?!(?:时候|话))"))
data = data[data.age <= 48]

print(f"The full dataset contains {len(data)} utterances between {data.age.min()} and {data.age.max()} months.")
data.to_csv("full_data.csv")
print("Data has been saved to a CSV file.")
#modifier_types = data["preceding_item_type"].value_counts()
#print(type(modifier_types))
#head_types = data["succeeding_item_type"].value_counts()
#print(head_types)
#variety = data["preceding_item_type"].nunique()
#print(variety)
