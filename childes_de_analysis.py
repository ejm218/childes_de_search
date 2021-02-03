import nltk
from nltk.corpus.reader import CHILDESCorpusReader
import numpy as np
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
data = create_df(child_utterances_data)
print(f"Dataframe successfully generated. There are {len(data)} rows and {len(data.columns)} in this Dataframe.")

# CREATE SUBGROUPS FOR ANALYSIS this part doesn't work yet and IDK why, age variable always returns None
#under_4 = []
#over_4 = []
#for corpus in corpora:
#    corpus_files = list(corpus.fileids())
#    for file in corpus_files:
#        age = corpus.age(fileids=file, month=True)
#        age = age[0]
#        if age <= 48:
#            under_4.append(file)
#        else:
#            over_4.append(file)
