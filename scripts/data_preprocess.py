import os
import glob
from datasets import load_dataset

path = './data/fulltext/'
#path = './try_data/'

globber = os.path.join(path, '**/*.txt')
files = glob.glob(globber, recursive = True)
files.sort()

for file in files:
    if 'clean' in file:
        continue
    file_name = file[:-4]
    if os.path.exists(file_name+'_clean.txt'):
        continue
    
    print(f'Cleaning text: {file}')
    f = open(file, 'r')
    lines = f.readlines()
     
    # remove arxiv head
#     idx = lines.index('X\n', 0, 50)
#     lines = lines[idx+4:]
    
#     l = ['\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n']
#     i = 0
#     while i < len(l):
#         print(l)
#         if i > 0 and len(l[i]) < 3 and l[i-1][-1] == '\n':
#             if l[i][0] != 'A' and l[i][0] != 'I':
#                 l.pop(i)
#                 continue
#         i += 1
#     print(len(l))
    
    # remove arxiv head
    while len(lines) > 0 and len(lines[0]) < 3:
        lines.pop(0)
        
    pop_idx = []
    num_lines = len(lines)
    for i in range(num_lines):
        # remove references
        if 'References\n' in lines[i]:
            lines = lines[:i-1]
            break
        
        # remove formulas (lines that are too short)
        if i > 0 and len(lines[i]) < 3:
            if len(lines[i-1]) > 0 and (lines[i-1][-1] == ' ' or lines[i-1][-1] == '\n'):
                if lines[i][0] != 'A' and lines[i][0] != 'I':
                    pop_idx.append(i)
                    continue
                
        # combine lines within a paragraph
        if i < num_lines-1 and lines[i+1] != '\n':
            lines[i] = lines[i][:-1] + ' '
        
        # words separated by -
        if lines[i][-2:] == '- ':
            lines[i] = lines[i][:-2]
        
    
    num_pop = len(pop_idx)
    print(num_lines - num_pop)
    for i in range(num_pop-1):
        #print(pop_idx[i])
        lines.pop(pop_idx[i] - i)
    #print(lines)
            
    f.close()
    
    # output clean txt
    f = open(file_name+'_clean.txt', 'w')
    f.writelines(lines)
    f.close()

    
globber = os.path.join(path, '**/*_clean.txt')
files = glob.glob(globber, recursive = True)
    
dataset = load_dataset('text', data_files=globber)

print(dataset)
print(dataset['train'][300])