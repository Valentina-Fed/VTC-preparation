# -*- coding: utf-8 -*-

import os
import argparse
import datalad.api as dl
import pandas as pd


def parse_dict_children(dict_children):
    children = {k:[] for k in ['train', 'dev', 'test']}
    with open(dict_children) as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('train'):
                v = line.replace('train: ', '')
                children['train'].extend(v.split(','))
            if line.startswith('train'):
                v = line.replace('dev: ', '')
                children['dev'].extend(v.split(','))
            if line.startswith('train'):
                v = line.replace('test: ', '')
                children['test'].extend(v.split(','))
    return children


def distribution_from_dict():
    children = parse_dict_children(dict_children)
    for file in os.listdir(f'{corpus}/recordings/raw'):
        if file.endswith('.wav'):
            name = file.split('_')[1]
            if name in children['train']:
                dl.copy_file(f'{corpus}/recordings/raw/{file}', target_dir=f'{output}/train')
            elif name in children['test']:
                dl.copy_file(f'{corpus}/recordings/raw/{file}', target_dir=f'{output}/test')
            elif name in children['dev']:
                dl.copy_file(f'{corpus}/recordings/raw/{file}', target_dir=f'{output}/dev')


def closest_value(size_dev, sum_child_durations):
    abs_diff_function = lambda list_value: abs(list_value - size_dev)
    closest_value = min(sum_child_durations.values(), key=abs_diff_function)
    return closest_value


def distribution():
    rec = f'{corpus}/metadata/recordings.csv'
    dl.get(rec)
    df = pd.read_csv(rec)
    child_id = df['child_id']
    recording_filename = df['recording_filename']
    duration = df['duration']
    total_duration = sum(duration)
    size_dev = total_duration // 5
    set_child = set(child_id)
    sum_child_durations = dict.fromkeys(set_child, 0)
    for i, child in enumerate(child_id):
        sum_child_durations[child] += duration[i]
    size = 0
    dict_children = dict.fromkeys(['train', 'dev', 'test'], [])
    while size < size_dev:
        val = closest_value(size_dev, sum_child_durations)
        size = size + val
        keys = [k for k, v in sum_child_durations.items() if v == val]
        dict_children['dev'].append(keys[0])
        sum_child_durations.pop(keys[0], None)
    size = 0
    while size < size_dev:
        val = closest_value(size_dev, sum_child_durations)
        size = size + val
        keys = [k for k, v in sum_child_durations.items() if v == val]
        dict_children['test'].append(keys[0])
        sum_child_durations.pop(keys[0], None)
    dict_children['train'] = sum_child_durations.keys()
    for k, v in dict_children.items():
        for i, child in enumerate(child_id):
            if child in v:
                if recording_filename[i] != 'NA' and type(recording_filename[i]) == str:
                    dl.copy_file(f'{corpus}/recordings/raw/{recording_filename[i]}', f'{output}/{k}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',
                      required=True,
                      help='the whole path to the folder where you will take the data of your corpus: /...'
                      )
    parser.add_argument('--output',
                      required=True,
                      help='the whole path to the folder yoda where you would like to store your output: /...'
                      )
    parser.add_argument('--dict_children',
                      required=False,
                      help='distribution dictionary'
                      )

    args = parser.parse_args()
    output = args.output
    corpus = args.corpus
    dict_children = args.dict_children
    m = corpus.split('/')
    name_corpus = (m[len(m) - 1]).replace('.git', '')

    df = pd.read_csv(f'{corpus}/metadata/annotations.csv')
    rec_name = df['recording_filename']
    onset = df['range_onset']
    offset = df['range_offset']
    dataset = df['set']
    filename = df['raw_filename']
    if dict_children:
        distribution_from_dict()
    else:
        distribution()



