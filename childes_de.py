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
data.loc[data["succeeding_item_type"] == "co", "succeeding_item_type"] = "det"
data.loc[data["succeeding_item_type"] == "co:iNP", "succeeding_item_type"] = "NA"
data.loc[data["succeeding_item_type"] == "coNP", "succeeding_item_type"] = "NA"
data.loc[data["succeeding_item_type"] == "asp", "succeeding_item_type"] = "NA"
data.loc[data["succeeding_item_type"] == "quaNP", "succeeding_item_type"] = "adj"
print(f"The full dataset contains {len(data)} utterances between {data.age.min()} and {data.age.max()} months.")
#modifier_types = data["preceding_item_type"].value_counts()
#print(type(modifier_types))
#head_types = data["succeeding_item_type"].value_counts()
#print(head_types)
#variety = data["preceding_item_type"].nunique()
#print(variety)

# SUBSETS
tong_data = data[data.filename.str.contains("Tong.*")==True]
print(f"There are {len(tong_data)} de utterances in the Tong dataset.}")
erbaugh_data = data[data.filename.str.contains("Erbaugh.*")==True]
print(f"There are {len(erbaugh_data)} de utterances in the Erbaugh dataset.}")
zhou1_data = data[data.filename.str.contains("Zhou1.*")==True]
print(f"There are {len(zhou1_data)} de utterances in the Zhou 1 dataset.}")
zhou3_data = data[data.filename.str.contains("Zhou3.*")==True]
print(f"There are {len(zhou3_data)} de utterances in the Zhou 3 dataset.}")
data_24to30 = pd.concat([tong_data, erbaugh_data, zhou3_data])
data_24to30 = data_24to30[data_24to30.age >= 24]
data_24to30 = data_24to30[data_24to30.age <= 30]
print(f"There are {len(data_24to30)} de utterances in the 24-30 month dataset.}")
data_36to48 = data[data.age >= 36]
data_36to48 = data_36to48[data_36to48.age <= 48]
print(f"There are {len(data_36to48)} de utterances in the 36-48 month dataset.}")

# CREATE FREQUENCY DATASETS
count_data = pd.DataFrame(columns=["age", "total_de", [data.succeeding_item_type.tolist()], [data.preceding_item_type.tolist()]])

tong_head_count_data = pd.crosstab(index=tong_data["age"], columns=tong_data["succeeding_item_type"], normalize=True)
tong_spec_count_data = pd.crosstab(index=tong_data["age"], columns=tong_data["preceding_item_type"], normalize=True)
print(f"Created DF tong_head_count_data with {len(tong_head_count_data)} rows.")
print(f"Created DF tong_spec_count_data with {len(tong_spec_count_data)} rows.")
erbaugh_head_count_data = pd.crosstab(index=erbaugh_data["age"], columns=erbaugh_data["succeeding_item_type"], normalize=True)
erbaugh_spec_count_data = pd.crosstab(index=erbaugh_data["age"], columns=erbaugh_data["preceding_item_type"], normalize=True)
print(f"Created DF erbaugh_head_count_data with {len(erbaugh_head_count_data)} rows.")
print(f"Created DF erbaugh_spec_count_data with {len(erbaugh_spec_count_data)} rows.")
zhou3_head_count_data = pd.crosstab(index=zhou3_data["age"], columns=zhou3_data["succeeding_item_type"], normalize=True)
zhou3_spec_count_data = pd.crosstab(index=zhou3_data["age"], columns=zhou3_data["preceding_item_type"], normalize=True)
print(f"Created DF zhou3_head_count_data with {len(zhou3_head_count_data)} rows.")
print(f"Created DF zhou3_spec_count_data with {len(zhou3_spec_count_data)} rows.")
zhou1_head_count_data = pd.crosstab(index=zhou1_data["age"], columns=zhou1_data["succeeding_item_type"], normalize=True)
zhou1_spec_count_data = pd.crosstab(index=zhou1_data["age"], columns=zhou1_data["preceding_item_type"], normalize=True)
print(f"Created DF zhou1_head_count_data with {len(zhou1_head_count_data)} rows.")
print(f"Created DF zhou1_head_count_data with {len(zhou1_head_count_data)} rows.")
etz_head_count_data = pd.crosstab(index=data_24to30["age"], columns=data_24to30["succeeding_item_type"], normalize=True)
etz_spec_count_data = pd.crosstab(index=data_24to30["age"], columns=data_24to30["preceding_item_type"], normalize=True)
print(f"Created DF etz_head_count_data with {len(etz_head_count_data)} rows. This is a combination of the Erbaugh, Tong, and Zhou3 dataframes, comprising data from {etz_count_data.age.min()} to {etz_count_data.age.max()} months.")
print(f"Created DF etz_spec_count_data with {len(etz_spec_count_data)} rows. This is a combination of the Erbaugh, Tong, and Zhou3 dataframes, comprising data from {etz_count_data.age.min()} to {etz_count_data.age.max()} months.")
older_head_count_data = pd.crosstab(index=data_36to48["age"], columns=data_36to48["succeeding_item_type"], normalize=True)
older_spec_count_data = pd.crosstab(index=data_36to48["age"], columns=data_36to48["preceding_item_type"], normalize=True)
print(f"Created DF older_head_count_data with {len(older_head_count_data)} rows. This is a combination of all relevant corpora, with data from {older_count_data.age.min()} to {older_count_data.age.max()} months.")
print(f"Created DF older_spec_count_data with {len(older_spec_count_data)} rows. This is a combination of all relevant corpora, with data from {older_count_data.age.min()} to {older_count_data.age.max()} months.")

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

#CREATING PLOT(s)
print("Now generating plots...")
# comparing preceding item types by ages
fig, axes = plt.subplots(nrows=2, ncols=2)
plt.
count_data.plot("age", ["sentence_final", "vp_head", "np_head"], kind="bar")
plt.savefig("headtype_raw.png")
count_data.plot("age", "total_de", kind="scatter")
plt.savefig("total_de_raw.png")

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
