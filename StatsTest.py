
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
from statsmodels.stats.weightstats import ztest

kgs = ['FB15k-237','wn18rr']#
models_ours = ['Butterfly', 'Butterfly_bias', 'EllipsE', 'EllipsE_Var']
models_sota = ['RotatE']#, 'ComplEx', 'TransE', 'DistMult']# , 'ComplEx'

file = open('stats_tests.txt', 'w')

#### Wilcoxon Signed-Rank Test
stat_kgs = []
for kg in kgs:
    head_data = pd.DataFrame(columns= models_ours + models_sota)
    tail_data = pd.DataFrame(columns= models_ours + models_sota)
    both_data = pd.DataFrame(columns= models_ours + models_sota)
    file.write('<'*10 + ' '*5 + kg + ' '*5 + '>'*20 +'\n')
    for sota in models_sota:
        file.write('-'*10 + ' '*5 + sota +'\n')
        stat_sota = []
        fileName = 'triples_ranked_' + kg + sota
        main_file = pd.read_csv(fileName +'.csv')
        head_data[sota] = main_file['head-batch']
        tail_data[sota] = main_file['tail-batch']
        both_data[sota] = main_file['ranks']
        print(f' {fileName} has length {main_file.shape[0]}')

        for ours in models_ours:
            fileName = 'triples_ranked_' + kg + ours
            main_file = pd.read_csv(fileName +'.csv')
            head_data[ours] = main_file['head-batch']
            tail_data[ours] = main_file['tail-batch']
            both_data[ours] = main_file['ranks']
            file.write('_'*5 + ours + '\n')
            print(f' {fileName} has length {main_file.shape[0]}')
            ## Wilcoxon
            # file.write('Wilcoxon Signed-Rank Test\n')
            # stat, p = wilcoxon(head_data[ours], head_data[sota])
            # print('stat=%.3f, p=%.3f' % (stat, p))
            # results = f'SAME probably distribution' if p > 0.05 else f'DIFFERENT probably distribution'
            # file.write(f'  head-mode: {results} with p = {p} and stat ={stat}\n\n')
            # stat, p = wilcoxon(tail_data[ours], tail_data[sota])
            # print('stat=%.3f, p=%.3f' % (stat, p))
            # results = f'SAME probably distribution' if p > 0.05 else f'DIFFERENT probably distribution'
            # file.write(f'  tail-mode: {results} with p = {p} and stat ={stat}\n\n')
            # stat, p = wilcoxon(both_data[ours], both_data[sota])
            # print('stat=%.3f, p=%.3f' % (stat, p))
            # results = f'SAME probably distribution' if p > 0.05 else f'DIFFERENT probably distribution'
            # file.write(f'  both-mode: {results} with p = {p} and stat ={stat}\n\n')

            ## Z Test
            file.write('Z-Test\n')
            stat, p = ztest(head_data[ours], head_data[sota])
            print('stat=%.3f, p=%.3f' % (stat, p))
            results = f'SAME probably distribution' if p > 0.05 else f'DIFFERENT probably distribution'
            file.write(f'  head-mode: {results} with p = {p} and stat ={stat}\n\n')
            stat, p = ztest(tail_data[ours]- tail_data[sota])
            print('stat=%.3f, p=%.3f' % (stat, p))
            results = f'SAME probably distribution' if p > 0.05 else f'DIFFERENT probably distribution'
            file.write(f'  tail-mode: {results} with p = {p} and stat ={stat}\n\n')
            stat, p = ztest(both_data[ours], both_data[sota])
            print('stat=%.3f, p=%.3f' % (stat, p))
            results = f'SAME probably distribution' if p > 0.05 else f'DIFFERENT probably distribution'
            file.write(f'  both-mode: {results} with p = {p} and stat ={stat}\n\n')

file.close()

