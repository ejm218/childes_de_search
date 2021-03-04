import nltk
import re
from nltk.corpus.reader import CHILDESCorpusReader
from collections import Counter
import pandas as pd
corpus_root = nltk.data.find("corpora/childes/data-xml/Mandarin") # change if your corpora are in a different path

#erbaugh = CHILDESCorpusReader(corpus_root, "Erbaugh/.*.xml")

def corpus_search(corpus_name, search_term, target_speaker="CHI", include_prior_utt=False):
    """Searches an entire corpus with parameters corpus_name (required, search_term (required), target_speaker (optional, default = CHI), and the option to include the preceding utterance (default = False). Search term can be a single word/phrase or a regular expression."""
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
            full_sentence = ""
            for word in sentence:
                full_sentence = full_sentence + " " + word
            if re.match(search_term, full_sentence):
                sentence_index = corpus_name.sents(transcript).index(sentence)
                corpus_search_results["filename"].append(transcript)
                corpus_search_results["age"].append(age)
                corpus_search_results["full_utterance"].append(sentence)
                if include_prior_utt == True:
                    corpus_search_results["preceding_utterance"] = []
                    corpus_search_results["preceding_utterance"].append(corpus_name.sents(transcript)[(int(sentence_index - 1))])
    print(f"The target speaker(s) {target_speaker} uses {search_term} {len(corpus_search_results)} total times.")
    return corpus_search_results

#test_frame = corpus_search(erbaugh, "妈妈", include_prior_utt=True)
#print(test_frame["preceding_utterance"])

# POS TAGGED WORDS ARE CONTAINED IN TUPLES. THIS CONVERTS THE TUPLE FOR EACH WORD+TAG TO A STRING
def tuple_to_string(tup):
    new_str = ""
    for i in tup:
        new_str = "|" + i
    return new_str

def corpus_search_MOR(corpus_name, search_term, target_speaker="CHI", include_prior_utt=False):
    """Searches an entire corpus with parameters corpus_name (required, search_term (required), target_speaker (optional, default = CHI), and the option to include the preceding utterance (default = False). Search term can be a single word/phrase or a regular expression. Each word and its corresponding POS tag are represented in the format word|tag, e.g. "at|prep". Keep this in mind if you are trying to search for both a word and its POS, and format your regex accordingly."""
    corpus_search_results = {
        "filename": [],
        "age": [],
        "full_utterance": []
        }
    transcripts = corpus_name.fileids()
    for transcript in transcripts:
        sentences = corpus_name.tagged_sents(transcript, speaker=target_speaker) # each sentence is a list of tuples
        age = corpus_name.age(fileids=transcript, month=True)
        age = age[0]
        for sentence in sentences:
            for item in sentence: # each word in the sentence is contained in a tuple where the first item is the word and the second is its MOR tag
                converted_item = tuple_to_string(item)
                if re.match(search_term, converted_item):
                    sentence_index = corpus_name.sents(transcript).index(sentence)
                    corpus_search_results["full_utterance"].append(sentence)
                    corpus_search_results["filename"].append(transcript)
                    corpus_search_results["age"].append(age)
                    if include_prior_utt == True:
                        corpus_search_results["preceding_utterance"] = []
                        corpus_search_results["preceding_utterance"].append(corpus_name.sents(transcript)[(int(sentence_index - 1))])
    return corpus_search_results

# THIS FUNCTION CREATES A DATAFRAME IN PANDAS FROM A GIVEN DICTIONARY (e.g. the one created with the function defined above)
def create_df(source_dict):
    column_names = source_dict.keys()
    df = pd.DataFrame(source_dict, columns=column_names)
    return df
