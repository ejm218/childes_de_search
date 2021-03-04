# THIS FILE CONTAINS THE FUNCTIONS USED TO GENERATE SEVERAL DIFFERENT DATASETS IN THE MAIN ANALYSIS
import nltk
from nltk.corpus.reader import CHILDESCorpusReader
from collections import Counter
import pandas as pd
corpus_root = nltk.data.find("corpora/childes/data-xml/Mandarin") # change if your corpora are in a different path
search_term = "的" # change this if you would like to search for a different lexical item

def corpus_search(corpus_name, search_term, target_speaker="CHI", include_prior_utt=False):
    """Searches an entire corpus with parameters corpus_name (required, search_term (required), target_speaker (optional, default = CHI), and the option to include the preceding utterance (default = False)."""
    corpus_search_results = {
        "filename": [],
        "age": [],
        "full_utterance": []
        }
    transcripts = corpus_name.fileids()
    for transcript in transcripts: # for each file within the given corpus...
        target_speaker = target_speaker
        age = corpus_name.age(fileids=transcript, month=True)
        sentences = corpus_name.sents(transcript, speaker=target_speaker)
        for sentence in sentences: # ...for each sentence in the file...
            if search_term in sentence:
                sentence_index = corpus_name.sents(transcript).index(sentence)
                corpus_search_results["filename"].append(transcript)
                corpus_search_results["age"].append(age)
                corpus_search_results["full_utterance"].append(sentence)
                if include_prior_utt == True:
                    corpus_search_results["preceding_utterance"] = []
                    corpus_search_results["preceding_utterance"].append(corpus_name.sents(transcript)[(sentence_index - 1)])
    print(f"The target speaker(s) {target_speaker} uses {search_term} {len(corpus_search_results)} total times.")
    return corpus_search_results

erbaugh = CHILDESCorpusReader(corpus_root, "Erbaugh/.*.xml")
test_frame = corpus_search(erbaugh, "妈妈", include_prior_utt=True)
print(test_frame["preceding_utterance"])

# POS TAGGED WORDS ARE CONTAINED IN TUPLES. THIS CONVERTS THE TUPLE FOR EACH WORD+TAG TO A STRING
def tuple_to_string(tup):
    new_str = ""
    for i in tup:
        new_str += i
    return new_str

# THIS FUNCTION EXTRACTS RELEVANT DATA FROM EACH TRANSCRIPT IN A CORPUS AND WRITES THE VALUES TO THE DICT DEFINED ABOVE
def generate_data(corpus_name):
    transcripts = corpus_name.fileids()
    people = corpus_name.participants(transcripts)
    chi_de_not_final = []
    filenames = []
    ages = []
    preceding_items = []
    succeeding_items = []
    full_utterances = []

# THIS FUNCTION CREATES A DATAFRAME IN PANDAS FROM A GIVEN DICTIONARY (e.g. the one created with the function defined above)
def create_df(source_dict):
    column_names = source_dict.keys()
    df = pd.DataFrame(source_dict, columns=column_names)
    return df

# SEARCH EXPLICITLY FOR MULTILEVEL RECURSIVE EMBEDDING
recursion_data = {
        "filename": [],
        "age": [],
        "full_utterance": [],
    }
def recursion_search(corpus_name, speaker="CHI"):
    transcripts = corpus_name.fileids()
    for transcript in transcripts:
        sentences = corpus_name.sents(transcript, speaker=speaker) # not using tagged sentences here to make it easier to use regex
        age = corpus_name.age(fileids=transcript, month=True)
        age = age[0]
        for sentence in sentences:
            full_sentence = ""
            for word in sentence:
                full_sentence = full_sentence + word
            if re.match(".+的[^时候].+的[^时候].+", full_sentence):
                recursion_data["filename"].append(transcript)
                recursion_data["age"].append(age)
                recursion_data["full_utterance"].append(full_sentence)
