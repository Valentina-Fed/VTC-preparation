import os
import pandas as pd
import shutil
from pathlib import Path
import datalad.api as dl
import argparse
from functools import partial
import librosa
import multiprocessing as mp
from os.path import (
    join as opj,
    basename,
    exists,
    splitext
)
import pandas as pd
import soundfile

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
                        help='dictionnary of distribution: "train", "dev", "test" with a list of children ids as a value'
                        )
    
    args = parser.parse_args()
    output = args.output
    path_corpus = args.corpus
    dict_children = args.dict_children
    m = path_corpus.split('/')
    name_corpus = (m[len(m) - 1]).replace('.git', '')


def change_directory(path):
    try:
        os.chdir(path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))


#path_corpus = 'git@gin.g-node.org:/LAAC-LSCP/tsay.git'

def move_files(source, target, files, substring):
    if files:
        for filename in Path(source).glob('*.*'):
            if substring in str(filename):
                shutil.copy(filename, target)

    else:
        alldirs = os.listdir(source)
        for f in alldirs:
            shutil.move(source + f, target + f)

change_directory(f'{output}/{name_corpus}/metadata/')
#list_children = dictionnary with 'train', 'dev', 'test'.
dict_children = {'train': ['LYC', 'HYS', 'HBL', 'LWJ', 'LMC'], 'dev': ['TWX'], 'test': ['CEY']}

def distribution_dict (dict_children): 
  rec = f'recordings.csv'
  dl.get(rec)
  df = pd.read_csv(rec)
  child_id = df['child_id']
  recording_filename = df['recording_filename']
  duration = df['duration']
  for k, v in dict_children.items():
    for i, child in enumerate(child_id):
      if child in v:
        dl.copy_file(f'{output}/{name_corpus}/recordings/raw/{recording_filename[i]}', f'{output}/input/{k}')
        dl.get(f'{output}/input/{k}/{recording_filename[i]}')
        if recording_filename[i].endswith('.wav'):
          ann_filename = recording_filename[i].replace('.wav', '')
          dl.copy_file(f'{output}/{name_corpus}/annotations/cha/converted/{ann_filename}_0_{duration[i]}.csv', f'{output}/input/{k}')
          dl.get(f'{output}/input/{k}/{ann_filename}_0_{duration[i]}.csv')
        elif recording_filename[i].endswith('.mp3'):
          ann_filename = recording_filename[i].replace('.mp3', '')
          dl.copy_file(f'{output}/{name_corpus}/annotations/cha/converted/{ann_filename}_0_{duration[i]}.csv', f'{output}/input/{k}')
          dl.get(f'{output}/input/{k}/{ann_filename}_0_{duration[i]}.csv')
distribution_dict(dict_children)




'''
if dict_children != None:
  distribution_dict(dict_children)
else:
  distribution()'''