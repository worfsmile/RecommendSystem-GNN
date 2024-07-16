
from flask import Flask, request, jsonify
import torch
import json
import random
import pandas as pd
import torch.nn.functional as F
from rec_book_depended_on_search import *

def find_median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 1:
        median = sorted_lst[n // 2]
    else:
        median = (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
    return median

def recommand_by_rating():
    with open('./data/merged_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    for name in list(data.keys()):
        data[name]["name"] = name
    data = list(data.values())
    name_list = [item['name'] for item in data]
    coe_list = [int(item['coe'].split(' ')[0]) for item in data]
    rating_list = [item['rating'] for item in data]
    # mean = sum(coe_list) / len(coe_list)
    middle = find_median(coe_list)
    rating_list = [(rating_list[i]*coe_list[i] / (coe_list[i]+middle), i) for i in range(len(name_list))]
    sorted_list = sorted(rating_list, key=lambda x: x[0])
    sorted_list.reverse()
    res = []
    for i in range(len(sorted_list)):
        res.append(name_list[sorted_list[i][1]])
    return res

def recommand_by_select(select_list):
    res = recommand_by_rating()
    select = select_books(select_list[0], select_list[1], select_list[2])
    result_list = list(set(res) - set(select_list))
    return result_list
  
def recommand(feature, feature_index, embed, index2book):
    out = F.cosine_similarity(feature, embed.unsqueeze(0), dim=2)
    def recommend(score):
        score_with_index = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
        result = [index for index, _ in score_with_index]
        # print(min(result), max(result))
        res = [book for book in result if book not in feature_index]
        return res
    score = out[0].tolist()
    recommend_indices = recommend(score)
    recommend_books = [index2book[str(i)] for i in recommend_indices]
    return recommend_books

def main(all_data_file="./data/merged_data.json", embed_path="./data/all_embeddings_GAT_32batch.pt", index2book_path="./data/index2book.json", rec_n = 10, book_names=None, book_indexs=None):
    data_file = all_data_file
    try:
        with open(data_file, 'r') as file:
            all_data = json.load(file)
    except:
        with open(data_file, 'r', encoding='utf-8') as file:
            all_data = json.load(file)
    if book_indexs==None and book_names==None:
        return recommand_by_rating()[:rec_n]
    embed = torch.load(embed_path)
    n = len(embed)
    if n < rec_n:
        rec_n = n

    with open(index2book_path, 'r') as file:
        index2book = json.load(file)
    book2index = {v: int(k) for k, v in index2book.items()}
    namelist = list(book2index.keys())
    book_n = len(namelist)
    if book_indexs:
        book_names = [index2book[str(name)] for name in book_indexs]
    else:
        book_indexs = [book2index[name] for name in book_names]
    print("历史阅读:", book_names)
    book_features = embed[torch.tensor(book_indexs, dtype=torch.long)]
    # print(book_features.shape)
    mean_values = book_features.mean(dim=0)
    # print(mean_values.shape)
    norm = torch.norm(mean_values, p=2)
    if norm.item() != 0:
        normalized_mean_values = mean_values / norm
    else:
        normalized_mean_values = mean_values
    feature = normalized_mean_values.unsqueeze(0)
    # print(feature)
    # print(feature)
    print("为你推荐:")
    if any(dim == 0 for dim in feature.shape):
        return recommand_by_rating()[:rec_n]
    else:
        return recommand(feature, book_indexs, embed, index2book)[:rec_n]
    
if __name__ == "__main__":
    print(main(book_indexs=[555,3333,11111]))
    print(main(book_names=['天机十二宫1：诡局']))
    print(main())