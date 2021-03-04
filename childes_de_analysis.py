import nltk
from nltk.corpus.reader import CHILDESCorpusReader
import numpy as np
import re
corpus_root = nltk.data.find("corpora/childes/data-xml/Mandarin") # be sure to change this if your corpora are located in a different path

# IMPORT THE CORPORA FOR USE IN THE SEARCH
chang1 = CHILDESCorpusReader(corpus_root, "Chang1/.*.xml")
chang2 = CHILDESCorpusReader(corpus_root, "Chang2/.*.xml")
changplay = CHILDESCorpusReader(corpus_root, "ChangPlay/.*.xml")
changpn = CHILDESCorpusReader(corpus_root, "ChangPN/.*.xml")
erbaugh = CHILDESCorpusReader(corpus_root, "Erbaugh/.*.xml")
lizhou = CHILDESCorpusReader(corpus_root, "LiZhou/.*.xml")
tong = CHILDESCorpusReader(corpus_root, "Tong/.*.xml")
zhou1 = CHILDESCorpusReader(corpus_root, "Zhou1/.*.xml")
zhou2 = CHILDESCorpusReader(corpus_root, "Zhou2/.*.xml")
zhou3 = CHILDESCorpusReader(corpus_root, "Zhou3/.*.xml")
corpora = [chang1, chang2, changplay, changpn, erbaugh, lizhou, tong, zhou1, zhou2, zhou3]

# PRINTS A GENERAL OVERVIEW OF DATA
print("Gathering all the corpora files...")
files_sum = 0
total_sentences = 0
total_num_tagged = 0
for i in corpora:
    #print(f"There are {len(i.fileids())} files in {i}")
    files_sum += len(i.fileids())
    sentences = len(i.sents())
    total_sentences += sentences
    num_tagged = len(i.tagged_sents())
    total_num_tagged += num_tagged
total_tagged = total_num_tagged / total_sentences
print(f"There are {files_sum} files to be evaluated in the list of corpora.")
print(f"{(total_tagged)*100}% of the target files have been MOR tagged.")

# CREATING VARIABLES FOR THE TRANSCRIPTS
chang1_transcripts = chang1.fileids()
chang2_transcripts = chang2.fileids()
changplay_transcripts = changplay.fileids()
changpn_transcripts = changpn.fileids()
erbaugh_transcripts = erbaugh.fileids()
lizhou_transcripts = lizhou.fileids()
tong_transcripts = tong.fileids()
zhou1_transcripts = zhou1.fileids()
zhou2_transcripts = zhou2.fileids()
zhou3_transcripts = zhou3.fileids()

# IMPORT THE FUNCTIONS GENERATE_DATA, REPORT, AND CREATE_DF
print("Importing special functions...")
functions_file = open("de_report.py", "r")
functions_file = functions_file.read()
exec(functions_file)

# CREATE A DF CONTAINING ALL INFO FROM ALL CORPORA
print("Creating a Dataframe...")
for corpus in corpora:
    generate_data(corpus) # this returns a dict object called child_utterances_data
full_data = create_df(child_utterances_data) # this saves the dict object to the variable called 'data'
print(f"Dataframe successfully generated. There are {len(full_data)} rows and {len(full_data.columns)} columns in this Dataframe.")

# CREATE SUBSETS OF THE DATA BASED ON AGE ETC
data_under_4 = full_data.copy()[full_data["age"] <= 48]
print(f"There are {len(data_under_4)} 的 utterances by children 48 months and younger.")
data_over_4 = full_data.copy()[full_data["age"] > 48]
print(f"There are {len(data_over_4)} 的 utterances by children older than 48 months.")

# REGEX TO FIND DIFFERENT POS TAGS
np_search = "n:?|pro:?" # finds all nouns and pronouns
vp_search = "v:?" # finds all verbs
adj_search = "adj" # finds all adjectives
possessives_data = full_data[full_data["preceding_item"].str.contains(np_search, na=False)]
print(f"There are {len(possessives_data)} rows in the possessives frame")
adj_data = full_data[full_data["preceding_item"].str.contains(adj_search, na=False)]
print(f"There are {len(adj_data)} rows in the adjectives frame")
relc_data = full_data[full_data["preceding_item"].str.contains(vp_search, na=False)]
print(f"There are {len(relc_data)} rows in the relative clauses frame")

# RECURSION
spreadsheet_name = "child_recursion.xlsx"
for corpus in corpora:
    recursion_search(corpus)
recursion_df = create_df(recursion_data)
print(f"There are {len(recursion_df)} possible instances of recursion in the dataset.")
recursion_df.to_excel(spreadsheet_name)
print(f"A spreadsheet has been created containing all possible recursive utterances.")
for value in recursion_data.values():
    value.clear() # otherwise the CDS search will simply add on to the pre-existing child speech data dict
print("Now searching for recursion in CDS...")
for corpus in corpora:
    adult_participants = []
    corpus_participants = corpus.participants(corpus.fileids())
    for this_corpus_participants in corpus_participants:
        for key in sorted(this_corpus_participants.keys()):
            adult_participants.append(key)
    adult_participants = list(set(adult_participants))
    adult_participants.remove("CHI")
    recursion_search(corpus, speaker=adult_participants)
cds_recursion_df = create_df(recursion_data)
print(f"Found {len(cds_recursion_df)} possible instances of recursion in child-directed speech.")
cds_recursion_df.to_excel("cds_recursion.xlsx")
print("A spreadsheet with these results has been created.")

# MANIPULATING THE RECURSION DATA
# after going through the output spreadsheet by hand
child_to_keep = [2,7,13,23,27,31,22,33,35,45,52,55,57,61,89,108,112,123,131,132]
recursion_df_culled = recursion_df.loc[child_to_keep]
recursion_df_culled["age"].value_counts()
recursion_df_culled = recursion_df_culled.loc[recursion_df_culled["age"] != 30]
recursion_df_culled = recursion_df_culled.loc[recursion_df_culled["age"] != 70] # these two were not recursive upon further inspection
recursion_over_48 = recursion_df_culled.loc[recursion_df_culled["age"] > 48]
recursion_under_48 = recursion_df_culled.loc[recursion_df_culled["age"] <= 48]
