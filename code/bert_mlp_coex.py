import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from collections import Counter
from functools import reduce
import operator
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
from scipy.spatial.distance import cdist
import spacy  
 
NUM_SAMPLES = 5000
NUM_BASE_CONCEPTS = 1000
EMBEDDING_NEIGHBORS = 5
EMBEDDING_PATH = 'glove.6B.50d.txt'
BATCH_SIZE = 8
CHUNK_SIZE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# From PENN-TREEBANK
PTB_TAGS = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", 
    "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", 
    "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", 
    "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", 
    "WP", "WP$", "WRB"
]

ds = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').eval().to(device)

texts = ds['train']['text'][:NUM_SAMPLES]

nlp = spacy.load("en_core_web_sm")

pos_assignments = []
for text in texts:
    doc = nlp(text)
    pos_tags_for_this_text = [token.tag_ for token in doc]
    pos_assignments.append(pos_tags_for_this_text)

max_len = 0
all_input_ids = []
for text in texts:
    encoded = tokenizer(text, padding=False, truncation=True, return_tensors='pt')
    input_ids = encoded['input_ids']
    seq_len = len(input_ids.squeeze(0)[1:-1])
    if seq_len > max_len:
        max_len = seq_len
    all_input_ids.append(input_ids)

def pad_sequence(arr, max_len):
    length = len(arr)
    if length < max_len:
        arr = np.concatenate([arr, np.zeros(max_len - length, dtype=arr.dtype)])
    return arr

def load_embeddings(path):
    vecs = []
    stoi = {}
    itos = {}
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            vals = np.array(list(map(float, parts[1:])), dtype=np.float32)
            stoi[tok] = idx
            itos[idx] = tok
            vecs.append(vals)
    vecs = np.stack(vecs, axis=0)
    return vecs, stoi, itos

VECS, VECS_STOI, VECS_ITOS = load_embeddings(EMBEDDING_PATH)

def get_neighbors(word, k=5):
    if word not in VECS_STOI:
        return []
    wid = VECS_STOI[word]
    wvec = VECS[wid:wid+1]
    dists = cdist(wvec, VECS, metric='cosine')[0]
    nearest_idx = np.argsort(dists)[1:k+1]
    return [VECS_ITOS[i] for i in nearest_idx]

counter = Counter()
for text in texts:
    tokens_ = text.split()
    counter.update(tokens_)
most_common = counter.most_common(NUM_BASE_CONCEPTS)
base_concept_tokens = [t for t,_ in most_common]

extended_concept_tokens = set(base_concept_tokens)
for t in base_concept_tokens:
    nbrs = get_neighbors(t, k=EMBEDDING_NEIGHBORS)
    extended_concept_tokens.update(nbrs)

extended_concept_tokens = list(extended_concept_tokens)

pos_token_prefix = "[POS_"
pos_tokens = [pos_token_prefix + tag + "]" for tag in PTB_TAGS]

pos_offset = len(extended_concept_tokens) 
extended_concept_tokens.extend(pos_tokens)

concept_ids = tokenizer.convert_tokens_to_ids(extended_concept_tokens)

concept_types = ['token'] * pos_offset + ['pos'] * len(pos_tokens)
assert len(concept_types) == len(extended_concept_tokens)
assert len(concept_ids)   == len(extended_concept_tokens)

tokens = extended_concept_tokens
num_concepts = len(concept_ids)
num_samples = len(all_input_ids)

def generate_concept_masks_in_chunks(
    all_input_ids, 
    concept_ids, 
    concept_types, 
    max_len, 
    pos_assignments,
    chunk_size=CHUNK_SIZE
):

    num_concepts = len(concept_ids)
    num_samples = len(all_input_ids)
    
    for chunk_start in range(0, num_concepts, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_concepts)
        chunk_concept_ids   = concept_ids[chunk_start:chunk_end]
        chunk_concept_types = concept_types[chunk_start:chunk_end]
        
        chunk_masks = np.zeros((len(chunk_concept_ids), max_len, num_samples), dtype=bool)
        
        for sample_idx, input_ids in enumerate(all_input_ids):
            arr = input_ids.squeeze(0)[1:-1].numpy()
            seq_len = len(arr)
            arr = pad_sequence(arr, max_len)

            sample_pos_tags = pos_assignments[sample_idx]
            if len(sample_pos_tags) < seq_len:
                sample_pos_tags += [""] * (seq_len - len(sample_pos_tags))
            sample_pos_tags = sample_pos_tags[:seq_len]
            
            for c_idx, cid in enumerate(chunk_concept_ids):
                concept_type = chunk_concept_types[c_idx]
                
                if concept_type == 'token':
                    chunk_masks[c_idx, :, sample_idx] = (arr == cid)
                
                elif concept_type == 'pos':
                    global_concept_idx = chunk_start + c_idx
                    pos_str = tokens[global_concept_idx]  
                    actual_tag = pos_str.replace(pos_token_prefix, "").replace("]", "")
                    
                    pos_mask = np.zeros(max_len, dtype=bool)
                    for t in range(seq_len):
                        if sample_pos_tags[t] == actual_tag:
                            pos_mask[t] = True
                    
                    chunk_masks[c_idx, :, sample_idx] = pos_mask
                
                else:
                    raise ValueError(f"Unknown concept type: {concept_type}")
        
        np.save(f'concept_masks_chunk_{chunk_start}.npy', chunk_masks)
        del chunk_masks
        torch.cuda.empty_cache()

generate_concept_masks_in_chunks(
    all_input_ids=all_input_ids, 
    concept_ids=concept_ids, 
    concept_types=concept_types,
    max_len=max_len, 
    pos_assignments=pos_assignments,
    chunk_size=CHUNK_SIZE
)

@dataclass
class Composition:
    formula: str
    mask: np.ndarray
    iou: float
    used_tokens: frozenset
    def __lt__(self, other):
        return self.iou < other.iou

def calculate_iou(pred: np.ndarray, target: np.ndarray):
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    return intersection / (union + 1e-10)

def apply_operation(op: str, mask1: np.ndarray, mask2: np.ndarray, invert_second: bool = False):
    if invert_second:
        mask2 = np.logical_not(mask2)
    if op == 'AND':
        return mask1 & mask2
    elif op == 'OR':
        return mask1 | mask2
    else:
        raise ValueError(f"Unknown op {op}")

def binarize_activation_mask(activation_mask, data_tokenized, percentile=50):
    if percentile == -1:
        threshold = 0
    else:
        threshold = np.percentile(activation_mask, percentile)
    return (activation_mask >= threshold) & (np.array(data_tokenized).T > 0)

def register_hook_for_location(model, layer_idx, activation_storage):
    def get_activation(name):
        def hook(m, inp, out):
            activation_storage[name] = out
        return hook
    h = model.encoder.layer[layer_idx].intermediate.intermediate_act_fn.register_forward_hook(
        get_activation(f'intermediate_{layer_idx}')
    )
    return h

def generate_activation_masks(dataset, location, model, tokenizer, length=5000, batch_size=BATCH_SIZE):
    layer, neuron = location
    texts_local = dataset['train']['text'][:length]
    activation_matrix = np.zeros((max_len, length))
    activation = {}
    hook_handle = register_hook_for_location(model, layer, activation)
    
    data_tokenized = []
    for text in texts_local:
        encoded = tokenizer(text, padding=False, truncation=True, return_tensors='pt')
        arr = encoded['input_ids'].squeeze(0)[1:-1].numpy()
        arr = pad_sequence(arr, max_len)
        data_tokenized.append(arr)
    
    for i in tqdm(range(0, len(data_tokenized), batch_size), desc="Generating activations"):
        batch = data_tokenized[i:i+batch_size]
        batch_tensor_list = []
        for arr in batch:
            full_seq = np.concatenate([[tokenizer.cls_token_id], arr, [tokenizer.sep_token_id]])
            batch_tensor_list.append(full_seq)
        
        batch_tensor = np.array(batch_tensor_list)
        batch_tensor = torch.tensor(batch_tensor, dtype=torch.long, device=device)
        
        with torch.no_grad():
            _ = model(batch_tensor)

        intermediate_output = activation[f'intermediate_{layer}']
        out_np = intermediate_output.detach().cpu().numpy()
        neuron_act = out_np[:, 1:-1, neuron]  # ignoring CLS, SEP

        for j in range(len(batch)):
            activation_matrix[:, i+j] = neuron_act[j]
        
        del intermediate_output, out_np, neuron_act
        torch.cuda.empty_cache()
    
    hook_handle.remove()
    return activation_matrix, np.array(data_tokenized)

def beam_search_composition(activation_mask, tokens, beam_width=10, max_depth=5, min_iou_increase=0.01):
    activation_mask = activation_mask.astype(bool)
    beam = []
    
    for chunk_start in range(0, len(tokens), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[chunk_start:chunk_end]
        
        chunk_file = f'concept_masks_chunk_{chunk_start}.npy'
        concept_masks = np.load(chunk_file).astype(bool)
        
        for i, tkn in enumerate(chunk_tokens):
            iou = calculate_iou(concept_masks[i], activation_mask)
            beam.append(Composition(tkn, concept_masks[i], iou, frozenset([tkn])))        

        del concept_masks
        torch.cuda.empty_cache()
    
    if not beam:
        return []
        
    beam = sorted(beam, key=lambda x: x.iou, reverse=True)[:beam_width]
    best_compositions = beam.copy()
    
    for depth in range(2, max_depth+1):
        prev_best_iou = max(c.iou for c in beam)
        candidates = []

        for comp in beam:
            used = comp.used_tokens
            for chunk_start in range(0, len(tokens), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(tokens))
                chunk_tokens = tokens[chunk_start:chunk_end]
                
                chunk_file = f'concept_masks_chunk_{chunk_start}.npy'
                concept_masks = np.load(chunk_file).astype(bool)
                
                for i, tkn in enumerate(chunk_tokens):
                    if tkn in used:
                        continue

                    ops_to_try = [
                        ('AND', False),
                        ('AND', True),
                        ('OR',  False),
                        ('OR',  True),
                    ]
                    for op_name, invert_second in ops_to_try:
                        new_mask = apply_operation(op_name, comp.mask, concept_masks[i], invert_second)
                        new_iou = calculate_iou(new_mask, activation_mask)
                        if invert_second:
                            new_formula = f"({comp.formula} {op_name} NOT {tkn})"
                        else:
                            new_formula = f"({comp.formula} {op_name} {tkn})"

                        candidates.append(
                            Composition(
                                formula=new_formula,
                                mask=new_mask,
                                iou=new_iou,
                                used_tokens=used.union([tkn])
                            )
                        )
                del concept_masks
                torch.cuda.empty_cache()
        
        if not candidates:
            break
        
        candidates = sorted(candidates, key=lambda x: x.iou, reverse=True)[:beam_width]
        new_best_iou = candidates[0].iou
        
        if (new_best_iou - prev_best_iou) < min_iou_increase:
            break

        beam = candidates
        best_compositions.extend(candidates)
    
    best_compositions = sorted(best_compositions, key=lambda x: x.iou, reverse=True)
    seen = set()
    unique_compositions = []
    for comp in best_compositions:
        if comp.formula not in seen:
            unique_compositions.append(comp)
            seen.add(comp.formula)

    return unique_compositions

results = []
layer = 11
num_neurons = 512
percentile = -1 

for neuron in range(num_neurons):
    act_masks, data_tokenized = generate_activation_masks(ds, (layer, neuron), model, tokenizer, length=NUM_SAMPLES)
    binary_act_mask = binarize_activation_mask(act_masks, data_tokenized, percentile=percentile)

    compositions = beam_search_composition(
        activation_mask=binary_act_mask,
        tokens=tokens,
        beam_width=5,
        max_depth=5,
        min_iou_increase=0.01
    )
    
    if compositions:
        best_comp = compositions[0]
        results.append({'layer': layer, 'neuron': neuron, 'formula': best_comp.formula, 'iou': best_comp.iou})
    else:
        results.append({'layer': layer, 'neuron': neuron, 'formula': 'No valid composition found', 'iou': 0.0})
    
    df = pd.DataFrame(results)
    df.to_csv(f'neuron_compositions_layer_{layer}_p{percentile}.csv', index=False)
    
    del act_masks, binary_act_mask, compositions
    torch.cuda.empty_cache()

df = pd.DataFrame(results)
print(df.nlargest(2, 'iou')[['neuron', 'formula', 'iou']])
