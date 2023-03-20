import numpy as np
import pandas as pd

data = pd.read_excel(r'C:\Users\bipro\OneDrive\Documents\Machine Learning\practical\F S algorithm dataset.xlsx')

concepts = np.array(data)[:,:-1]
target = np.array(data)[:,-1]

# Initialize h to the most specific hypothesis in H
def train(con, tar):
    for i,val in enumerate(tar):
        if val == 'yes':
            specific_h = con[i].copy()
            break;
    
    for i,val in enumerate(con):
        if[tar]=='yes':
            for x in range(len(specific_h)):
                if val[x] !=specific_h[x]:
                    specific_h[x]='?'
                else:
                    pass
    return specific_h

print(train(concepts, target))
