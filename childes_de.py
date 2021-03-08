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

# SUBSETS
tong_data = data[data.filename.str.contains("Tong.*")==True]
print(f"There are {len(tong_data)} de utterances in the Tong dataset.}")
erbaugh_data = data[data.filename.str.contains("Erbaugh.*")==True]
print(f"There are {len(erbaugh_data)} de utterances in the Erbaugh dataset.}")
zhou3_data = data[data.filename.str.contains("Zhou3.*")==True]
print(f"There are {len(zhou3_data)} de utterances in the Zhou 3 dataset.}")
data_24to30 = pd.concat([tong_data, erbaugh_data, zhou3_data])
data_24to30 = data_24to30[data_24to30.age >= 24]
data_24to30 = data_24to30[data_24to30.age <= 30]
print(f"There are {len(data_24to30)} de utterances in the 24-30 month dataset.}")
data_36to48 = data[data.age >= 36]
data_36to48 = data_36to48[data_36to48.age <= 48]
print(f"There are {len(data_36to48)} de utterances in the 36-48 month dataset.}")

# INTERIM SUMMARY OF DataFrame
print(f"DATA SUMMARY \n Age range: {data['age'].min()}-{data['age'].max()} months\n Number of head types: {data['succeeding_item_type'].nunique()}\n There are {len(data)} rows in the dataframe.\n Now printing first five rows: ")
#modifier_types = data["preceding_item_type"].value_counts()
#print(type(modifier_types))
#head_types = data["succeeding_item_type"].value_counts()
#print(head_types)
#variety = data["preceding_item_type"].nunique()
#print(variety)
print(data.head())

# CREATE FREQUENCY SUBSETS
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
count_data = create_df(count_data)
count_data = count_data.sort_values(by="age")
print(f"Created DF count_data with {len(count_data)} rows.")
tong_count_data = {
    "age": [],
    "total_de": [],
    "np_head": [],
    "vp_head": [],
    "sentence_final": [],
    "prec_adj": [],
    "prec_np": [],
    "prec_vp": []
    }
for age in ages:
    target_data = tong_data[(tong_data["age"] == age)]
    tong_count_data["age"].append(age)
    tong_count_data["total_de"].append(len(target_data))
    tong_count_data["np_head"].append(len(target_data[(target_data["succeeding_item_type"] == "NP")]))
    tong_count_data["vp_head"].append(len(target_data[(target_data["succeeding_item_type"] == "VP")]))
    tong_count_data["sentence_final"].append(len(target_data[(target_data["succeeding_item_type"] == "NA")]))
    tong_count_data["prec_adj"].append(len(target_data[(target_data["preceding_item_type"] == "adj")]))
    tong_count_data["prec_np"].append(len(target_data[(target_data["preceding_item_type"] == "NP")]))
    tong_count_data["prec_vp"].append(len(target_data[(target_data["preceding_item_type"] == "VP")]))
tong_count_data = create_df(tong_count_data)
tong_count_data = tong_count_data[tong_count_data.age <= 40]
tong_count_data = tong_count_data[tong_count_data.total_de > 0]
print(f"Created DF tong_count_data with {len(tong_count_data)} rows.")
erbaugh_count_data = {
    "age": [],
    "total_de": [],
    "np_head": [],
    "vp_head": [],
    "sentence_final": [],
    "prec_adj": [],
    "prec_np": [],
    "prec_vp": []
    }
for age in ages:
    target_data = erbaugh_data[(erbaugh_data["age"] == age)]
    erbaugh_count_data["age"].append(age)
    erbaugh_count_data["total_de"].append(len(target_data))
    erbaugh_count_data["np_head"].append(len(target_data[(target_data["succeeding_item_type"] == "NP")]))
    erbaugh_count_data["vp_head"].append(len(target_data[(target_data["succeeding_item_type"] == "VP")]))
    erbaugh_count_data["sentence_final"].append(len(target_data[(target_data["succeeding_item_type"] == "NA")]))
    erbaugh_count_data["prec_adj"].append(len(target_data[(target_data["preceding_item_type"] == "adj")]))
    erbaugh_count_data["prec_np"].append(len(target_data[(target_data["preceding_item_type"] == "NP")]))
    erbaugh_count_data["prec_vp"].append(len(target_data[(target_data["preceding_item_type"] == "VP")]))
erbaugh_count_data = create_df(erbaugh_count_data)
erbaugh_count_data = erbaugh_count_data[erbaugh_count_data.total_de > 0]
print(f"Created DF erbaugh_count_data with {len(erbaugh_count_data)} rows.")
zhou3_count_data = {
    "age": [],
    "total_de": [],
    "np_head": [],
    "vp_head": [],
    "sentence_final": [],
    "prec_adj": [],
    "prec_np": [],
    "prec_vp": []
    }
for age in ages:
    target_data = zhou3_data[(zhou3_data["age"] == age)]
    zhou3_count_data["age"].append(age)
    zhou3_count_data["total_de"].append(len(target_data))
    zhou3_count_data["np_head"].append(len(target_data[(target_data["succeeding_item_type"] == "NP")]))
    zhou3_count_data["vp_head"].append(len(target_data[(target_data["succeeding_item_type"] == "VP")]))
    zhou3_count_data["sentence_final"].append(len(target_data[(target_data["succeeding_item_type"] == "NA")]))
    zhou3_count_data["prec_adj"].append(len(target_data[(target_data["preceding_item_type"] == "adj")]))
    zhou3_count_data["prec_np"].append(len(target_data[(target_data["preceding_item_type"] == "NP")]))
    zhou3_count_data["prec_vp"].append(len(target_data[(target_data["preceding_item_type"] == "VP")]))
zhou3_count_data = create_df(zhou3_count_data)
zhou3_count_data = zhou3_count_data[zhou3_count_data.total_de > 0]
print(f"Created DF zhou3_count_data with {len(zhou3_count_data)} rows.")
etz_count_data = {
    "age": [],
    "total_de": [],
    "np_head": [],
    "vp_head": [],
    "sentence_final": [],
    "prec_adj": [],
    "prec_np": [],
    "prec_vp": []
    }
for age in ages:
    target_data = data_24to30[(data_24to30["age"] == age)]
    etz_count_data["age"].append(age)
    etz_count_data["total_de"].append(len(target_data))
    etz_count_data["np_head"].append(len(target_data[(target_data["succeeding_item_type"] == "NP")]))
    etz_count_data["vp_head"].append(len(target_data[(target_data["succeeding_item_type"] == "VP")]))
    etz_count_data["sentence_final"].append(len(target_data[(target_data["succeeding_item_type"] == "NA")]))
    etz_count_data["prec_adj"].append(len(target_data[(target_data["preceding_item_type"] == "adj")]))
    etz_count_data["prec_np"].append(len(target_data[(target_data["preceding_item_type"] == "NP")]))
    etz_count_data["prec_vp"].append(len(target_data[(target_data["preceding_item_type"] == "VP")]))
etz_count_data = create_df(etz_count_data)
etz_count_data = etz_count_data[etz_count_data.total_de > 0]
print(f"Created DF etz_count_data with {len(etz_count_data)} rows. This is a combination of the Erbaugh, Tong, and Zhou3 dataframes, comprising data from {etz_count_data.age.min()} to {etz_count_data.age.max()} months.")
older_count_data = {
    "age": [],
    "total_de": [],
    "np_head": [],
    "vp_head": [],
    "sentence_final": [],
    "prec_adj": [],
    "prec_np": [],
    "prec_vp": []
    }
for age in ages:
    target_data = data_36to48[(data_36to48["age"] == age)]
    older_count_data["age"].append(age)
    older_count_data["total_de"].append(len(target_data))
    older_count_data["np_head"].append(len(target_data[(target_data["succeeding_item_type"] == "NP")]))
    older_count_data["vp_head"].append(len(target_data[(target_data["succeeding_item_type"] == "VP")]))
    older_count_data["sentence_final"].append(len(target_data[(target_data["succeeding_item_type"] == "NA")]))
    older_count_data["prec_adj"].append(len(target_data[(target_data["preceding_item_type"] == "adj")]))
    older_count_data["prec_np"].append(len(target_data[(target_data["preceding_item_type"] == "NP")]))
    older_count_data["prec_vp"].append(len(target_data[(target_data["preceding_item_type"] == "VP")]))
older_count_data = create_df(older_count_data)
older_count_data = older_count_data[older_count_data.total_de > 0]
print(f"Created DF older_count_data with {len(older_count_data)} rows. This is a combination of all relevant corpora, with data from {older_count_data.age.min()} to {older_count_data.age.max()} months.")

#CREATING PLOT(s)
count_data.plot("age", ["sentence_final", "vp_head", "np_head"], kind="bar")
plt.savefig("headtype_raw.png")
count_data.plot("age", "total_de", kind="scatter")
plt.savefig("total_de_raw.png")

# create new DF where each column is represented as percentage of total de counts
count_data_normed = count_data.copy()
count_data_normed = count_data_normed.astype(float)
def get_percentages(dataframe, numerator, denominator):
    """Returns a new dataframe where values in one column (the numerator) is expressed as a percentage of values in another column (the denominator). The numerator can also be a list of column names."""
    df_normed = dataframe.copy()
    df_normed = df_normed.astype(float)
    for i in range(len(df_normed)):
        total_count = int(df_normed.at[i, denominator])
        for column in numerator:
            df_normed.at[i, column] = ((int(df_normed.at[i, column]) / total_count)*100)
    return df_normed
for i in range(len(count_data_normed)):
    total_count = int(count_data_normed.at[i, "total_de"])
    count_data_normed.at[i, "np_head"] = ((int(count_data_normed.at[i, "np_head"]) / total_count)*100)
    count_data_normed.at[i, "vp_head"] = ((int(count_data_normed.at[i, "vp_head"]) / total_count)*100)
    count_data_normed.at[i, "sentence_final"] = ((int(count_data_normed.at[i, "sentence_final"]) / total_count)*100)
    count_data_normed.at[i, "prec_adj"] = ((int(count_data_normed.at[i, "prec_adj"]) / total_count)*100)
    count_data_normed.at[i, "prec_np"] = ((int(count_data_normed.at[i, "prec_np"]) / total_count)*100)
    count_data_normed.at[i, "prec_vp"] = ((int(count_data_normed.at[i, "prec_vp"]) / total_count)*100)
for row in data:
    for age in ages:
        unique_nouns = data.prec_np[data.age == age].nunique()
        unique_nouns = data.prec_vp[data.age == age].nunique()
        unique_nouns = data.prec_adj[data.age == age].nunique()
        print(f"Age: {age} months\n Number of unique nouns before de: {unique_nouns}")

# MORE PLOTS
# comparing preceding item types by ages
count_data_normed.plot("age", ["prec_adj", "prec_np", "prec_vp"], kind="bar")
plt.xlabel("Age in months")
plt.ylabel("Proportion of ADJ, NP, and VP preceding DE")
plt.savefig("preceding_items.png")
# comparing head types by age
count_data_normed.plot("age", ["np_head", "vp_head"], kind="bar")
plt.xlabel("Age in months")
plt.ylabel("Proportion of NP and VP heads")
plt.savefig("preceding_items.png")
# scatter plot of VP heads over time
count_data_normed.plot("age", "vp_head", kind="scatter")
plt.xlabel("Age in months")
plt.ylabel("Proportion VP heads")
plt.savefig("vp_head_vs_age.png")

fig, axes = plt.subplots(nrows=2, ncols=2)
tong_count_data.plot(ax=axes[0,0])
plt.scatter(x=tong_count_data.age, y=tong_count_data.total_de)
plt.scatter(x=tong_count_data.age, y=tong_count_data.np_head, "^")
plt.scatter(x=tong_count_data.age, y=tong_count_data.vp_head, "s")
plt.scatter(x=tong_count_data.age, y=tong_count_data.sentence_final, "d")
classes = ["Total de utterances", "NP heads", "VP heads", "Sentence-final de"]
plt.xlabel("Age in months")
plt.ylabel("Number of utterances containing 'de'")
plt.legend(labels=classes)
plt.title("Usage of de in the Tong corpus")
zhou3_count_data.plot(ax=axes[0,1])
plt.scatter(x=zhou3_count_data.age, y=zhou3_count_data.total_de)
plt.scatter(x=zhou3_count_data.age, y=zhou3_count_data.np_head, "^")
plt.scatter(x=zhou3_count_data.age, y=zhou3_count_data.vp_head, "s")
plt.scatter(x=zhou3_count_data.age, y=zhou3_count_data.sentence_final, "d")
classes = ["Total de utterances", "NP heads", "VP heads", "Sentence-final de"]
plt.xlabel("Age in months")
plt.ylabel("Number of utterances containing 'de'")
plt.legend(labels=classes)
plt.title("Usage of de in the Zhou 3 corpus")
erbaugh_count_data.plot(ax=axes[1,0])
plt.scatter(x=erbaugh_count_data.age, y=erbaugh_count_data.total_de)
plt.scatter(x=erbaugh_count_data.age, y=erbaugh_count_data.np_head, "^")
plt.scatter(x=erbaugh_count_data.age, y=erbaugh_count_data.vp_head, "s")
plt.scatter(x=erbaugh_count_data.age, y=erbaugh_count_data.sentence_final, "d")
classes = ["Total de utterances", "NP heads", "VP heads", "Sentence-final de"]
plt.xlabel("Age in months")
plt.ylabel("Number of utterances containing 'de'")
plt.legend(labels=classes)
plt.title("Usage of de in the Erbaugh corpus")
etz_count_data.plot(ax=axes[1,1])
plt.scatter(x=etz_count_data.age, y=etz_count_data.total_de)
plt.scatter(x=etz_count_data.age, y=etz_count_data.np_head, "^")
plt.scatter(x=etz_count_data.age, y=etz_count_data.vp_head, "s")
plt.scatter(x=etz_count_data.age, y=etz_count_data.sentence_final, "d")
classes = ["Total de utterances", "NP heads", "VP heads", "Sentence-final de"]
plt.xlabel("Age in months")
plt.ylabel("Number of utterances containing 'de'")
plt.legend(labels=classes)
plt.title("Usage of de in the three combined corpora")
plt.savefig("de_dev_early.png")

# CALCULATE REGRESSION LINE AKA LINE OF BEST FIT
from statistics import mean
def get_regression_line(x ,y ):
    m = (((mean(x)*mean(y)) - mean(x*y)) /
         ((mean(x)*mean(x)) - mean(x*x)))
    b = mean(y) - m*mean(x)
    regression_line = [(m*value)+b for value in x]
    return regression_line
