# SMALLER SEARCH USING ONLY THE LIZHOU CORPUS (for practice)
import nltk
from nltk.corpus.reader import CHILDESCorpusReader
import numpy as np
corpus_root = nltk.data.find("corpora/childes/data-xml/Mandarin")
lizhou = CHILDESCorpusReader(corpus_root, "LiZhou/.*.xml")
transcripts = lizhou.fileids()
people = lizhou.participants(transcripts)
chi_sentences = []

# all child utterances
for transcript in transcripts:
    sentences = lizhou.sents(transcript, speaker="CHI")
    for sentence in sentences:
        chi_sentences.append(sentence)

# all child utterances containing 的
chi_de_sentences = []
for sentence in chi_sentences:
    if "的" in sentence:
        chi_de_sentences.append(sentence)

# all utterances containing 的 with something after it
chi_de_not_final = []
for sentence in chi_de_sentences:
    if sentence[-1] != "的":
        chi_de_not_final.append(sentence)
print(f"The child uses DE {len(chi_de_sentences)} total times, and {len(chi_de_not_final)} times in non-sentence final position.")

# show what words come before and after 的 in child utterances and count their occurrences
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
