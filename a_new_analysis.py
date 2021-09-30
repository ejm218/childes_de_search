# IMPORT DEPENDENCIES
import pandas as pd
import matplotlib as plt

# IMPORT FULL DATA SET

# CLEAN DATA
# collapse all nouns to NP
# collapse all verbs (except shi) to VP

# FAST FACTS
print(f"Total number of utterances: {len(df)}")
#age range
#most frequent age
#most frequent preceding item
#most frequent succeeding item
#avg utterance length?

# CREATE SUBSET "ETZ" CONTAINING TONG, ERBAUGH, AND ZHOU3 FROM 24-30MO
tong_data = data[data.filename.str.contains("Tong.*")==True]
print(f"There are {len(tong_data)} de utterances in the Tong dataset.")
erbaugh_data = data[data.filename.str.contains("Erbaugh.*")==True]
print(f"There are {len(erbaugh_data)} de utterances in the Erbaugh dataset.")
zhou3_data = data[data.filename.str.contains("Zhou3.*")==True]
print(f"There are {len(zhou3_data)} de utterances in the Zhou 3 dataset.")
etz_data = pd.concat([tong_data, erbaugh_data, zhou3_data])
etz_data = etz_data[data_24to30.age >= 24]
etz_data = etz_data[data_24to30.age <= 30]
print(f"There are {len(etz_data)} de utterances in the 24-30 month dataset.")

# YOUNGEST AGE: ZHOU3 BEGINNING 8MO
# number of utterances by age

# succeeding item type by age
zhou3_head_count_data = pd.crosstab(index=zhou3_data["age"], columns=zhou3_data["succeeding_item_type"], normalize="index").reset_index()# preceding item type by age
zhou3_head_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Proportion of Head types in 'de' utterances")
plt.title("Types of items after 'de' in the Zhou 3 corpus")
plt.savefig("zhou3_succeeding.png")
# preceding item type by age 
zhou3_spec_count_data = pd.crosstab(index=zhou3_data["age"], columns=zhou3_data["preceding_item_type"], normalize="index").reset_index()
zhou3_spec_count_data.plot.bar(x="age", stacked=True)
plt.xlabel("Age in months")
plt.ylabel("Proportion of Spec types in 'de' utterances")
plt.title("Types of items preceding 'de' in the Zhou 3 corpus")
plt.savefig("zhou3_preceding.png")

# CROSS-SECTIONAL: ZHOU 1,2 AND LIZHOU
lizhou_data = data[data.filename.str.contains("LiZhou.*")==True]
print(f"There are {len(lizhou_data)} de utterances in the LiZhou dataset.")
zhou1_data = data[data.filename.str.contains("Zhou1.*")==True]
print(f"There are {len(zhou1_data)} de utterances in the Zhou 1 dataset.")
zhou2_data = data[data.filename.str.contains("Zhou2.*")==True]
print(f"There are {len(zhou2_data)} de utterances in the Zhou 2 dataset.")
xsectional_data = pd.concat([lizhou_data, zhou1_data, zhou2_data])
print(f"There are {len(xsectional_data)} de utterances in the cross-sectional corpora.")

