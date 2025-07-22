from datasets import load_dataset

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003", split='train')

# Get tag mapping
ner_label_map = dataset.features["ner_tags"].feature.names
pos_label_map = dataset.features["pos_tags"].feature.names

# Save file to CoNLL format
def save_as_conll(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for example in dataset:
            for token, pos_id, ner_id in zip(example['tokens'], example['pos_tags'], example['ner_tags']):
                pos_tag = pos_label_map[pos_id]
                ner_tag = ner_label_map[ner_id]
                f.write(f"{token} {pos_tag} - {ner_tag}\n")
            f.write("\n")

save_as_conll(dataset, "data/conll2003_train.txt")