import json
import numpy as np
import copy

from datasets import load_dataset, Dataset, DatasetDict
from bs4 import BeautifulSoup


def load_acl(mode, sep_token):
    path = './data/aclarc/' + mode + '.jsonl'
    data = []
    data_lines = open(path, 'r')
    for line in data_lines:
        data.append(json.loads(line))
    text_data = []
    label_data = []
    to_label = {
        'Background': 0,
        'Extends': 1,
        'Uses': 2,
        'Motivation': 3,
        'CompareOrContrast': 4,
        'Future': 5,
    }
    
    for item in data:
        text_data.append('<acl> ' + item['text'])
        label_data.append(to_label[item['label']])
    dict_data = {
        'text': text_data,
        'label': label_data,
    }
    return dict_data


def load_scierc(mode, sep_token):
    path = './data/scierc/' + mode + '.jsonl'
    data = []
    data_lines = open(path, 'r')
    for line in data_lines:
        data.append(json.loads(line))
    text_data = []
    label_data = []
    to_label = {
        'COMPARE': 0,
        'PART-OF': 1,
        'CONJUNCTION': 2,
        'EVALUATE-FOR': 3,
        'FEATURE-OF': 4,
        'USED-FOR': 5,
        'HYPONYM-OF': 6,
    }
    
    for item in data:
        text_data.append('<sci> ' + item['text'])
        label_data.append(to_label[item['label']])
    dict_data = {
        'text': text_data,
        'label': label_data,
    }
    return dict_data


def load_sem(sep_token):
    # TBD
    text_data = []
    label_data = []
    to_label = {
        'COMPARE': 0,
        'MODEL-FEATURE': 1,
        'PART_WHOLE': 2,
        'RESULT': 3,
        'TOPIC': 4,
        'USAGE': 5,
    }
    
    text_path = '/code/project/data/semeval/test_text.xml'
    with open(file_path, 'r') as text_raw:
        raw_xml = text_raw.read()
    xml = BeautifulSoup(raw_xml, features="html.parser")
    texts = xml.doc.findAll('text')
    

    for text in texts:
        abstract_o = text.find_all('abstract')
        print(abstract_o)
        abstract = ' '.join([ii.text.strip() for ii in abstract_o])
        print(abstract)
        
        # Split abstract into sentences
        sentences = abstract.split('. ')
        for i in range(len(sentences)-1):
            sentences[i] += '.'
        print(sentences)

        # Store entities in dict
        entities = text.findAll('entity')
        entity = {}
        for e in entities:
            entity.update({e['id']: e.text})
        print(entity)
    # Todo: Check if entity is a substr of text
        
    # Find sentence-relation pairs
    with open(filename, 'r') as relation_raw:
        raw_lines = relation_raw.readlines()
    for string in raw_lines:
        pass

    
def build_fs(dict_train, data_name):
    train_text = dict_train['text']
    train_label = dict_train['label']
    
    label_num = 0
    if data_name == 'acl':
        label_num = 6
    if data_name == 'scierc':
        label_num = 7
    
    label_idx = [[]]*label_num
    for i in range(len(train_label)):
        label_idx[train_label[i]] = copy.copy(label_idx[train_label[i]])
        label_idx[train_label[i]].append(i)
    idx = []  
    
    for i in range(label_num):
        print(len(label_idx[i]))
        idx.extend(list(np.random.choice(label_idx[i], 8)))
    
    fs_text = [train_text[i] for i in idx]
    fs_label = [train_label[i] for i in idx]
    dict_fs = {
        'text': fs_text,
        'label': fs_label, 
    }
    return dict_fs


def get_dataset(dataset_name, seed=20, do_fs=False, sep_token='<sep>'):
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    # your code for preparing the dataset...
    
    np.random.seed(seed)
    
    if dataset_name == 'both':
        dataset_name = ['acl', 'scierc']
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    
    label_num = 0
    dict_test_acl = {}
    dict_test_sci = {}
    
    for data_name in dataset_name:
        # print(f'Building dataset: {name}...')
        
        # get dataset
        if data_name == 'scierc':
            dict_train = load_scierc('train', sep_token)
            dict_test = load_scierc('test', sep_token)
            dict_val = load_scierc('dev', sep_token)
            dict_test_sci = dict_test
        if data_name == 'acl':
            dict_train = load_acl('train', sep_token)
            dict_test = load_acl('test', sep_token)
            dict_val = load_acl('dev', sep_token)
            dict_test_acl = dict_test
        
        # build few-shot training set
        if do_fs:
            dict_train = build_fs(dict_train, data_name)
        
        # aggregate
        if label_num == 0:
            agg_train = dict_train
            agg_test = dict_test
            agg_val = dict_val
        else:
            dict_train['label'] = [i+label_num for i in dict_train['label']]
            dict_test['label'] = [i+label_num for i in dict_test['label']]
            dict_val['label'] = [i+label_num for i in dict_val['label']]
            for t in ['text', 'label']:
                agg_train[t].extend(dict_train[t])
                agg_test[t].extend(dict_test[t])
                agg_val[t].extend(dict_val[t])
        label_num += len(set(dict_train['label']))
        
        # log
        # train_size = len(dict_train['label'])
        # test_size = len(dict_test['label'])
        # print(f'Done: {name} built with training size {train_size}, test size {test_size}.')
    
    dataset = DatasetDict({
                'train': Dataset.from_dict(agg_train),
                'test': Dataset.from_dict(agg_test),
                'val': Dataset.from_dict(agg_val),
                'test_acl': Dataset.from_dict(dict_test_acl),
                'test_sci': Dataset.from_dict(dict_test_sci),
            })
    
    return dataset


def get_text(path, sep_token='<sep>'):
    #acl
    text_acl = load_acl('train', sep_token)
    text_acl = text_acl['text']
    text_acl = [text + '\n' for text in text_acl]
    f = open(path + '/acl.txt', 'w')
    f.writelines(text_acl)
    f.close()
    
    # sci
    text_sci = load_scierc('train', sep_token)
    text_sci = text_sci['text']
    text_sci = [text + '\n' for text in text_sci]
    f = open(path + '/sci.txt', 'w')
    f.writelines(text_sci)
    f.close()
    
    # both
    f = open(path + '/both.txt', 'w')
    f.writelines(text_acl + text_sci)
    f.close()
    
#get_text('/code/project/data')
        

    
# #dataset = get_dataset('SemEval', 101, False)
# dataset = load_dataset('/code/project/data/semeval/test_text.xml')
# print(dataset)
# #print(dataset['train'][0])

#load_sem('<sep>')



