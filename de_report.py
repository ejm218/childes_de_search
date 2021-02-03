# THIS FILE CONTAINS THE FUNCTIONS USED TO GENERATE A DF IN THE MAIN ANALYSIS
import nltk
from nltk.corpus.reader import CHILDESCorpusReader
from collections import Counter
import pandas as pd
corpus_root = nltk.data.find("corpora/childes/data-xml/Mandarin") # change if your corpora are in a different path
search_term = "的" # change this if you would like to search for a different lexical item

def report(corpus_name):
    transcripts = corpus_name.fileids()
    people = corpus_name.participants(transcripts)
    chi_sentences = []
    chi_de_sentences = []
    chi_de_not_final = []
    for transcript in transcripts: # for each file within the given corpus...
        target_speaker = "CHI" # change this if you are targeting child-directed speech
        sentences = corpus_name.sents(transcript, speaker=target_speaker)
        for sentence in sentences: # ...for each sentence in the file...
            chi_sentences.append(sentence) # ...add it to the variable containing the list of sentences
    for sentence in chi_sentences:
        if search_term in sentence:
            chi_de_sentences.append(sentence)
    for sentence in chi_de_sentences:
        if sentence[-1] != search_term:
            chi_de_not_final.append(sentence)
    print(f"The child uses DE {len(chi_de_sentences)} total times, {len(chi_de_not_final)} in non-sentence final position.")
    precedes_de = []
    succeeds_de = []
    for sentence in chi_de_not_final:
        de_index = sentence.index(search_term)
        before_index = int(de_index - 1)
        after_index = int(de_index + 1)
        precedes_de.append(sentence[before_index]) # creates a list of the items that appear immediately before de in the data
        succeeds_de.append(sentence[after_index]) # creates a list of the items that appear immediately after de
    from collections import Counter
    precedes_de_count = Counter(precedes_de)
    succeeds_de_count = Counter(succeeds_de)
    precedes_de_unique = set(precedes_de)
    succeeds_de_unique = set(succeeds_de)
    print(f"In the {len(chi_de_not_final)} non-sentence final utterances, there are {len(precedes_de_unique)} unique items that occur BEFORE 的, and {len(succeeds_de_unique)} unique items that occur AFTER 的.")

# CREATE A DICT ITEM THAT WE WILL WRITE TO USING GENERATE_DATA
child_utterances_data = {
        "filename": [],
        "age": [],
        "preceding_item": [],
        "succeeding_item": [],
        "full_utterance": [],
    }
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
    for transcript in transcripts:
        sentences = corpus_name.sents(transcript, speaker="CHI")
        age = corpus_name.age(fileids=transcript, month=True)
        age = age[0]
        for sentence in sentences:
            if "的" in sentence and sentence[-1] != search_term:
                chi_de_not_final.append(sentence)
                child_utterances_data["filename"].append(transcript)
                child_utterances_data["age"].append(age)
                de_index = sentence.index(search_term)
                before_index = int(de_index - 1)
                after_index = int(de_index + 1)
                child_utterances_data["preceding_item"].append(sentence[before_index])
                child_utterances_data["succeeding_item"].append(sentence[after_index])
                child_utterances_data["full_utterance"].append(sentence)

# THIS FUNCTION CREATES A DATAFRAME IN PANDAS FROM A GIVEN DICTIONARY (e.g. the one created above)
def create_df(source_dict):
    column_names = source_dict.keys()
    df = pd.DataFrame(source_dict, columns=column_names)
    return df
