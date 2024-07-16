
from flask import Flask, request, jsonify
import torch
import json
import random
import pandas as pd
import torch.nn.functional as F
from rec_book_depended_on_search import *

def recommand_by_select():
    with open("./data/index2book.json", 'r') as file:
        index2book = json.load(file)
    select_list = [["小说"], ["长", "中"], ['鲁迅']]
    res = sorted_by_rating()
    try:
        select = select_books(select_list[0], select_list[1], select_list[2])
        select = list(select)
        for i in range(len(select)):
            select[i] = index2book[str(select[i])]
        if len(select) < 3:
            return random.sample(res[:10], 3)
        return random.sample(select, 3)
    except:
        return random.sample(res[:10], 3)

def recommend_by_history():
    data_file="./data/merged_data.json"
    embed_path="./data/normalized_embeddings_GAT_128batch.pt"
    index2book_path="./data/index2book.json"
    rec_n = 10
    book_names=['天机十二宫1：诡局']
    try:
        with open(data_file, 'r') as file:
            all_data = json.load(file)
    except:
        with open(data_file, 'r', encoding='utf-8') as file:
            all_data = json.load(file)
    if book_names==None:
        res = sorted_by_rating()[:100]
        return  random.sample(res, 3)
    embed = torch.load(embed_path)
    n = len(embed)
    if n < rec_n:
        rec_n = n
    with open(index2book_path, 'r') as file:
        index2book = json.load(file)
    book2index = {v: int(k) for k, v in index2book.items()}
    namelist = list(book2index.keys())
    book_n = len(namelist)    
    book_indexs = [book2index[name] for name in book_names]
    print("历史阅读:", book_names)
    book_features = embed[torch.tensor(book_indexs, dtype=torch.long)]
    
    mean_values = book_features.mean(dim=0)
    norm = torch.norm(mean_values, p=2)
    if norm.item() != 0:
        normalized_mean_values = mean_values / norm
    else:
        normalized_mean_values = mean_values
    feature = normalized_mean_values.unsqueeze(0)
    
    print("为你推荐:")
    if any(dim == 0 for dim in feature.shape):
        return random.sample(sorted_by_rating()[:rec_n], 3)
    else:
        return random.sample(recommand(feature, book_indexs, embed, index2book)[:rec_n], 3)
    
if __name__ == "__main__":
    print(recommend_by_history())
    print(recommend_by_history())
    print(recommand_by_select())