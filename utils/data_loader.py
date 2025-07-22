import streamlit as st

# Utilities to load CoNLL-format data
def load_conll_data(filepath):
    sentences = []
    current = []
    
    with open(filepath, 'r', encoding='utf-8') as f:

        for line in f:
            if line.strip() == "": # Reset current sentence on empty line
                if current:
                    sentences.append(current)
                    current = []
            else: # Process each line, append to current sentence
                token, pos, _, ner = line.strip().split()
                current.append((token, pos, ner))
        if current:
            sentences.append(current)
    
    return sentences
