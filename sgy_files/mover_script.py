import os

path = os.path.join('..', '..', '..', 'exps', 'Lamb_demphire')

if not os.path.exists(path):
    raise FileNotFoundError

dir_name = 'demphire_coord_table_mm_0'

for dir in os.listdir(path):
    if not os.path.exists(os.path.join(path, dir, 'output')):
        print(dir)
        continue

    os.makedirs(os.path.join(dir_name, dir), exist_ok=True)
    os.makedirs(os.path.join(dir_name, dir, 'output'), exist_ok=True)
    
    for file in os.listdir(os.path.join(path, dir, 'output')):
        if file.endswith('.sgy') or file.endswith('.txt'):
            os.system(f"copy {os.path.join(path, dir, 'output', file)} {os.path.join(dir_name, dir, 'output', file)}")

