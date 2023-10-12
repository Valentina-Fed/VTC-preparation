import os
import pandas as pd
import pylangacq
import textgrid
import collections

#for parsing TextGrid annotations
def parse_textgrid(annotation, child, rec_name):
    start, duration, speaker = ([] for n in range(3))
    tg = textgrid.TextGrid.fromFile(annotation)
    dico = collections.defaultdict(list)
    for i in range(len(tg)):
        for j in range(len(tg[i])):
            if tg[i][j].mark != '' and tg[i][j].mark != ' ' and tg[i][j].mark != '0' and tg[i][j].minTime not in dico:
                v = (tg[i].name.strip(), '{0:.4f}'.format(round(tg[i][j].maxTime - tg[i][j].minTime, 4)))
                dico[tg[i][j].minTime - 180].append(v)
    for k, v in sorted(dico.items()):
        if v[0][0] == 'LF2P' or v[0][0] == 'Autre' or v[0][0] == '2POPMT' or v[0][0].startswith('Loin') or '2parl' in v[0][0]:
            continue
        else:
            start.append('{0:.4f}'.format(round(k, 4)))
            duration.append(v[0][1])
            if v[0][0] == 'CHI*':
                speaker.append(f'!CHI_{child}')
            else:
                speaker.append(v[0][0])
    rttm = {'SP': ['SPEAKER'] * len(speaker), 'rec': [rec_name] * len(speaker), 'nb': ['1'] * len(speaker),
            'start': start, 'duration': duration, 'na1': ['<NA>'] * len(speaker), 'na2': ['<NA>'] * len(speaker), 'speaker': speaker,
            'na3': ['<NA>'] * len(speaker), 'na4': ['<NA>'] * len(speaker)}
    df_rttm = pd.DataFrame(data=rttm)
    return df_rttm

def create_dataset(subset):
    return [name for name in os.listdir(f'{output}/{subset}') if name.endswith('.wav')]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',
                      required=True,
                      help='the whole path to the corpus: /...')
    parser.add_argument('--output',
                      required=True,
                      help='the whole path to the folder yoda where you would like to store your output: /...'
                      )

    args = parser.parse_args()
    output = args.output
    corpus = args.corpus
    n = corpus.split('/')
    o = output.split('/')
    name_corpus = n[-1]
    name_set = o[-1]
    dataset = ['train', 'dev', 'test']
    train, dev, test = map(create_dataset, dataset)
    whole_dataset = train + dev + test
    rttm_train, rttm_dev, rttm_test, uem_tr, uem_t, uem_d, dur_full_tr, dur_full_d, dur_full_t = ([] for n in range(9))
    corpora = ['vanuatu', 'namibia', 'tsimane2017']
    df = pd.read_csv(f'{corpus}/metadata/annotations.csv')

    if name_corpus == 'tsay':
        recording_filename = df['recording_filename']
        ann_filename = df['annotation_filename']
        duration_full = df['range_offset']
        INV_name = 'Rose'
        for i, rec in enumerate(recording_filename):
              if rec.endswith('.wav'):
                  start, duration, speaker = ([] for in range(3))
                  rec_tsay = rec.replace('Tsay', 'tsay')
                  rec_f = rec.split('_')
                  rec_ts = rec.replace('.wav', '')
                  rec_name = rec_tsay.replace('.wav', '')
                  data = pd.read_csv(f'{corpus}/annotations/cha/converted/{ann_filename[i]}')
                  speaker_id, onset, offset = data['speaker_id'], data['segment_onset'], offset = data['segment_offset']
                  child = pylangacq.read_chat(f'{corpus}/annotations/cha/raw/{rec_ts}.cha')
                  if 'INV' in child.participants():
                      if child.headers()[0]['Participants']['INV']['name'] != None:
                        INV_name = child.headers()[0]['Participants']['INV']['name']
                  elif 'IN1' in child.participants():
                      if child.headers()[0]['Participants']['IN1']['name'] != None:
                        INV_name = child.headers()[0]['Participants']['IN1']['name']
                  elif 'IN2' in child.participants():
                      if child.headers()[0]['Participants']['IN2']['name'] != None:
                        INV_name = child.headers()[0]['Participants']['IN2']['name']
                  for j, sp in enumerate(speaker_id):
                      if sp == 'LF2P' or sp == 'Autre' or sp == '2POPMT' or sp == 'Loin' or sp == '+2parl':
                        continue
                      else:
                        duration.append('{0:.4f}'.format((offset[j] - onset[j])/1000))
                        start.append('{0:.4f}'.format(onset[j]/1000))
                        if sp == 'INV':
                          speaker.append(f'!INV_{INV_name}')
                        elif sp == 'IN1':
                          speaker.append(f'!IN1_{INV_name}')
                        elif sp == 'IN2':
                          speaker.append(f'!IN2_{INV_name}')
                        elif sp == 'CHI' or sp == 'CHI*':
                          speaker.append(f'!CHI_{rec_f[1]}')
                        else:
                          speaker.append(sp)
                  rttm = {'SP': ['SPEAKER']*len(speaker), 'rec': [rec_name]*len(speaker), 'nb': ['1'] * len(speaker), 'start': start,
                      'duration': duration, 'na1': ['<NA>']*len(speaker), 'na2': ['<NA>']*len(speaker), 'speaker': speaker,
                      'na3': ['<NA>']*len(speaker), 'na4': ['<NA>']*len(speaker)}
                  df_rttm = pd.DataFrame(data=rttm)
                  nb = round(int(duration_full[i]) / 1000, 0)
                  if rec_tsay in train:
                        df_rttm.to_csv(f'/{output}/train/gold/{rec_name}.rttm', sep=' ', index=False, header=False)
                        rttm_train.append(df_rttm)
                        uem_tr.append(rec_tsay)
                        dur_full_tr.append('{0:.2f}'.format(nb))
                  elif rec_tsay in dev:
                        df_rttm.to_csv(f'/{output}/dev/gold/{rec_name}.rttm', sep=' ', index=False, header=False)
                        rttm_dev.append(df_rttm)
                        uem_d.append(rec_tsay)
                        dur_full_d.append('{0:.2f}'.format(nb))
                  elif rec_tsay in test:
                        df_rttm.to_csv(f'/{output}/test/gold/{rec_name}.rttm', sep=' ', index=False, header=False)
                        rttm_test.append(df_rttm)
                        uem_t.append(rec_tsay)
                        dur_full_t.append('{0:.2f}'.format(nb))
        rttm_tr = pd.concat(t for t in rttm_train)
        rttm_tr.to_csv(f'{output}/train/tsay.train.rttm', sep=' ', index=False, header=False)
        rttm_d = pd.concat(t for t in rttm_dev)
        rttm_d.to_csv(f'{output}/dev/tsay.dev.rttm', sep=' ', index=False, header=False)
        rttm_t = pd.concat(t for t in rttm_test)
        rttm_t.to_csv(f'{output}/test/tsay.test.rttm', sep=' ', index=False, header=False)
        uem_tr_pd = {'rec': uem_tr, 'nb': ['1']*len(uem_tr), 'onset': ['{0:.3f}'.format(0.000)]*len(uem_tr),
               'dur': dur_full_tr}
        uem_t_pd = {'rec': uem_t, 'nb': ['1']*len(uem_t), 'onset': ['{0:.3f}'.format(0.000)]*len(uem_t),
               'dur': dur_full_t}
        uem_d_pd = {'rec': uem_d, 'nb': ['1']*len(uem_d), 'onset': ['{0:.3f}'.format(0.000)]*len(uem_d),
               'dur': dur_full_d}
        uem_tr_df = pd.DataFrame(data=uem_tr_pd)
        uem_tr_df.to_csv(f'{output}/train/tsay.train.uem', sep=' ', index=False, header=False)
        uem_t_df = pd.DataFrame(data=uem_t_pd)
        uem_t_df.to_csv(f'{output}/test/tsay.test.uem', sep=' ', index=False, header=False)
        uem_d_df = pd.DataFrame(data=uem_d_pd)
        uem_d_df.to_csv(f'{output}/dev/tsay.dev.uem', sep=' ', index=False, header=False)

    elif name_corpus in corpora:
        dataset, recording_filename, ann_filename, offset, onset = df['set'], df['recording_filename'], df['raw_filename'], offset = df['range_offset'], onset = df['range_onset']
        for i, rec in enumerate(recording_filename):
            if dataset[i].startswith('textgrid') and rec.endswith('.wav'):
                  if name_corpus == 'tsimane2017':
                      child = (rec.split('_'))[1]
                      rec_name = f"{rec.replace('.wav', '')}_{onset[i]//1000}_{offset[i]//1000}"
                      ann = 'mm'
                  else:
                      rec1 = rec.split('/')
                      child = rec1[0]
                      rec_name = f"{rec1[1].replace('.wav', '')}_{onset[i]//1000}_{offset[i]//1000}"
                      ann1 = ann_filename[i].split('_')
                      ann = (ann1[-1]).replace('.TextGrid', '')
                  nb = round((offset[i]-onset[i]) / 1000, 0)
                  if f'{rec_name}.wav' in train:
                      df_rttm = parse_textgrid(f'{corpus}/annotations/textgrid/{ann}/raw/{ann_filename[i]}', child, rec_name)
                      df_rttm.to_csv(f'{output}/train/gold/{rec_name}.rttm', sep = ' ', index=False, header=False)
                      rttm_train.append(df_rttm)
                      uem_tr.append(rec_name)
                      dur_full_tr.append('{0:.2f}'.format(nb))
                  elif f'{rec_name}.wav' in dev:
                      df_rttm = parse_textgrid(f'{corpus}/annotations/textgrid/{ann}/raw/{ann_filename[i]}', child, rec_name)
                      df_rttm.to_csv(f'{output}/dev/gold/{rec_name}.rttm', sep = ' ', index=False, header=False)
                      rttm_dev.append(df_rttm)
                      uem_d.append(rec_name)
                      dur_full_d.append('{0:.2f}'.format(nb))
                  elif f'{rec_name}.wav' in test:
                      df_rttm = parse_textgrid(f'{corpus}/annotations/textgrid/{ann}/raw/{ann_filename[i]}', child, rec_name)
                      df_rttm.to_csv(f'{output}/test/gold/{rec_name}.rttm', sep = ' ', index=False, header=False)
                      rttm_test.append(df_rttm)
                      uem_t.append(rec_name)
                      dur_full_t.append('{0:.2f}'.format(nb))
        rttm_tr = pd.concat(t for t in rttm_train)
        rttm_tr.to_csv(f'{output}/train/{name_corpus}.train.rttm', sep=' ', index=False, header=False)
        rttm_d = pd.concat(t for t in rttm_dev)
        rttm_d.to_csv(f'{output}/dev/{name_corpus}.dev.rttm', sep=' ', index=False, header=False)
        rttm_t = pd.concat(t for t in rttm_test)
        rttm_t.to_csv(f'{output}/test/{name_corpus}.test.rttm', sep=' ', index=False, header=False)
        uem_tr_pd = {'rec': uem_tr, 'nb': ['1']*len(uem_tr), 'onset': ['{0:.3f}'.format(0.000)]*len(uem_tr),
               'dur': dur_full_tr}
        uem_t_pd = {'rec': uem_t, 'nb': ['1']*len(uem_t), 'onset': ['{0:.3f}'.format(0.000)]*len(uem_t),
               'dur': dur_full_t}
        uem_d_pd = {'rec': uem_d, 'nb': ['1']*len(uem_d), 'onset': ['{0:.3f}'.format(0.000)]*len(uem_d),
               'dur': dur_full_d}
        uem_tr_df = pd.DataFrame(data=uem_tr_pd)
        uem_tr_df.to_csv(f'{output}/train/{name_corpus}.train.uem', sep=' ', index=False, header=False)
        uem_t_df = pd.DataFrame(data=uem_t_pd)
        uem_t_df.to_csv(f'{output}/test/{name_corpus}.test.uem', sep=' ', index=False, header=False)
        uem_d_df = pd.DataFrame(data=uem_d_pd)
        uem_d_df.to_csv(f'{output}/dev/{name_corpus}.dev.uem', sep=' ', index=False, header=False)
    else:
        dataset = df['set']
        recording_filename = df['recording_filename']
        ann_filename = df['annotation_filename']
        offset = df['range_offset']
        onset = df['range_onset']
        for i, rec in enumerate(recording_filename):
            start = []
            duration = []
            if '/' in rec:
                rec1 = rec
                rec = rec1.split('/')[1]
            rec_name = rec.split('.')[0]
            if name_corpus == 'solomon':
                child1 = rec_name.split('_')
                child = '_'.join(child1[:5])
            else:
                child = rec_name.split('_')[0]
            recname = f'{name_corpus}_{rec_name}_{onset[i]}_{offset[i]}'
            if dataset[i].startswith('eaf') or dataset[i].startswith('cha'):
                df_ann = pd.read_csv(f'{corpus}/annotations/{dataset[i]}/converted/{ann_filename[i]}')
                segment_onset, segment_offset, speaker = df_ann['segment_onset'], df_ann['segment_offset'], df_ann['speaker_id']
                speaker_id=[]
                for j, sp in enumerate(speaker):
                    if sp == 'CHI' or sp == 'CHI*':
                        speaker_id.append(f'!CHI_{child}')
                    else:
                        speaker_id.append(sp)
                    duration.append('{0:.4f}'.format((segment_offset[j] - segment_onset[j]) / 1000))
                    start.append('{0:.4f}'.format(segment_onset[j] / 1000))
                rttm = {'SP': ['SPEAKER'] * len(speaker_id), 'rec': [recname] * len(speaker_id), 'nb': ['1'] * len(speaker_id),
                      'start': start, 'duration': duration, 'na1': ['<NA>'] * len(speaker_id), 'na2': ['<NA>'] * len(speaker_id),
                      'speaker': speaker_id, 'na3': ['<NA>'] * len(speaker_id), 'na4': ['<NA>'] * len(speaker_id)}
                df_rttm = pd.DataFrame(data=rttm)
                if onset[i] == 0:
                    nb = round(int(offset[i]) / 1000, 0)
                else:
                    nb = round((offset[i] - onset[i]) / 1000, 0)
                    if (f'{recname}.wav') in train:
                        df_rttm.to_csv(f'/{output}/train/gold/{recname}.rttm', sep=' ', index=False, header=False)
                        rttm_train.append(df_rttm)
                        uem_tr.append(recname)
                        dur_full_tr.append('{0:.2f}'.format(nb))
                    elif (f'{recname}.wav') in dev:
                        df_rttm.to_csv(f'/{output}/dev/gold/{recname}.rttm', sep=' ', index=False, header=False)
                        rttm_dev.append(df_rttm)
                        uem_d.append(recname)
                        dur_full_d.append('{0:.2f}'.format(nb))
                    elif f'{recname}.wav' in test:
                        df_rttm.to_csv(f'/{output}/test/gold/{recname}.rttm', sep=' ', index=False, header=False)
                        rttm_test.append(df_rttm)
                        uem_t.append(recname)
                        dur_full_t.append('{0:.2f}'.format(nb))
        rttm_tr = pd.concat(t for t in rttm_train)
        rttm_tr.to_csv(f'{output}/train/{name_corpus}.train.rttm', sep=' ', index=False, header=False)
        rttm_d = pd.concat(t for t in rttm_dev)
        rttm_d.to_csv(f'{output}/dev/{name_corpus}.dev.rttm', sep=' ', index=False, header=False)
        rttm_t = pd.concat(t for t in rttm_test)
        rttm_t.to_csv(f'{output}/test/{name_corpus}.test.rttm', sep=' ', index=False, header=False)
        uem_tr_pd = {'rec': uem_tr, 'nb': ['1'] * len(uem_tr), 'onset': ['{0:.3f}'.format(0.000)] * len(uem_tr),
                       'dur': dur_full_tr}
        uem_t_pd = {'rec': uem_t, 'nb': ['1'] * len(uem_t), 'onset': ['{0:.3f}'.format(0.000)] * len(uem_t),
                      'dur': dur_full_t}
        uem_d_pd = {'rec': uem_d, 'nb': ['1'] * len(uem_d), 'onset': ['{0:.3f}'.format(0.000)] * len(uem_d),
                      'dur': dur_full_d}
        uem_tr_df = pd.DataFrame(data=uem_tr_pd)
        uem_tr_df.to_csv(f'{output}/train/{name_corpus}.train.uem', sep=' ', index=False, header=False)
        uem_t_df = pd.DataFrame(data=uem_t_pd)
        uem_t_df.to_csv(f'{output}/test/{name_corpus}.test.uem', sep=' ', index=False, header=False)
        uem_d_df = pd.DataFrame(data=uem_d_pd)
        uem_d_df.to_csv(f'{output}/dev/{name_corpus}.dev.uem', sep=' ', index=False, header=False)



















