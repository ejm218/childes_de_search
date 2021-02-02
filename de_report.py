import nltk
from nltk.corpus.reader import CHILDESCorpusReader
from collections import Counter
import pandas as pd
corpus_root = nltk.data.find("corpora/childes/data-xml/Mandarin")
lizhou = CHILDESCorpusReader(corpus_root, "LiZhou/.*.xml")
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


def report(corpus_name):
    transcripts = corpus_name.fileids()
    people = corpus_name.participants(transcripts)
    chi_sentences = []
    chi_de_sentences = []
    chi_de_not_final = []
    for transcript in transcripts:
        sentences = corpus_name.sents(transcript, speaker="CHI")
        for sentence in sentences:
            chi_sentences.append(sentence)
    for sentence in chi_sentences:
        if "的" in sentence:
            chi_de_sentences.append(sentence)
    for sentence in chi_de_sentences:
        if sentence[-1] != "的":
            chi_de_not_final.append(sentence)
    print(f"The child uses DE {len(chi_de_sentences)} total times, {len(chi_de_not_final)} in non-sentence final position.")
    precedes_de = []
    succeeds_de = []
    for sentence in chi_de_not_final:
        de_index = sentence.index("的")
        before_index = int(de_index - 1)
        after_index = int(de_index + 1)
        precedes_de.append(sentence[before_index])
        succeeds_de.append(sentence[after_index])
    from collections import Counter
    precedes_de_count = Counter(precedes_de)
    succeeds_de_count = Counter(succeeds_de)
    precedes_de_unique = set(precedes_de)
    succeeds_de_unique = set(succeeds_de)
    print(f"In the {len(chi_de_not_final)} non-sentence final utterances, there are {len(precedes_de_unique)} unique items that occur BEFORE 的, and {len(succeeds_de_unique)} unique items that occur AFTER 的.")

child_utterances_data = {
        "filename": [],
        "age": [],
        "preceding_item": [],
        "succeeding_item": [],
        "full_utterance": [],
    }

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
            if "的" in sentence and sentence[-1] != "的":
                chi_de_not_final.append(sentence)
                child_utterances_data["filename"].append(transcript)
                child_utterances_data["age"].append(age)
                de_index = sentence.index("的")
                before_index = int(de_index - 1)
                after_index = int(de_index + 1)
                child_utterances_data["preceding_item"].append(sentence[before_index])
                child_utterances_data["succeeding_item"].append(sentence[after_index])
                child_utterances_data["full_utterance"].append(sentence)

def create_df(source_dict):
    column_names = source_dict.keys()
    df = pd.DataFrame(source_dict, columns=column_names)
    return df
