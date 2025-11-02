import os

files_list = os.listdir('./generated/')

dirs_path = os.path.join('..', '..', '..', '..', 'exps', 'Lamb_demphire')
if not os.path.exists(dirs_path):
    raise FileNotFoundError

for file in files_list:
    if file.endswith('.jou'):
        os.makedirs(os.path.join(dirs_path, file[:-4]), exist_ok=True)