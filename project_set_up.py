import os
   
data_path = 'data/'
subfolders = [
    'columns', 
    'input', 
    'models', 
    'pipelines', 
]

for f in subfolders:
    directory = data_path + f
    if not os.path.exists(directory):
        os.makedirs(directory)