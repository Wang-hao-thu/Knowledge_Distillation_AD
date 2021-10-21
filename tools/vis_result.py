import sys

import numpy as np
from tqdm import tqdm

result_file = sys.argv[1]
def get_result(result_file):
    f1 = open(result_file,'r')
    neg = {}
    neg_score = []
    pos = {}
    pos1_score = []
    pos2_score = []
    for line in tqdm(f1.readlines()):
        img_path, label, score = line.strip().split(' ')
        if int(label) == 0:
            neg_score.append(float(score))
            neg.update({str(score):img_path})
        elif int(label) == 1:
            pos1_score.append(float(score))
            pos.update({str(score): img_path})
        elif int(label) == 2:
            pos2_score.append(float(score))
    neg_score = np.array(neg_score)
    neg_score = sorted(neg_score,reverse=True)
    pos1_score = np.array(pos1_score)
    for rate in [0.5,0.2,0.1,0.05,0.01,0.005,0.001]:
        threshold = neg_score[int(rate * len(neg_score))]
        recalled_1 = sum(pos1_score>threshold)
        print(f"fp:{rate:.5f} ({int(rate * len(neg_score))}/{len(neg_score)}) , \
        recall1: {recalled_1/len(pos1_score):.5f} ({recalled_1}/{len(pos1_score)}, \
        threshold:{threshold:.5f}")
    threshold = neg_score[1]
    recalled_1 = sum(pos1_score>threshold)
    #print(neg[str(threshold)])

    print(f"fp:{rate:.5f} ({int(rate * len(neg_score))+1}/{len(neg_score)}) , \
            recall1: {recalled_1/len(pos1_score):.5f} ({recalled_1}/{len(pos1_score)}, \
            threshold:{threshold:.5f}",flush=True)


def main():
    get_result(result_file)

if __name__=='__main__':
    main()
