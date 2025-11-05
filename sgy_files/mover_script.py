import os

path = os.path.join('..', '..', '..', 'exps', 'Lamb_demphire')

if not os.path.exists(path):
    raise FileNotFoundError

for dir in os.listdir(path):
    os.makedirs(os.path.join('demphire_coord_table', dir), exist_ok=True)
    os.makedirs(os.path.join('demphire_coord_table', dir, 'output'), exist_ok=True)

    for file in os.listdir(os.path.join(path, dir, 'output')):
        if file.endswith('.sgy') or file.endswith('.txt'):
            os.system(f"copy {os.path.join(path, dir, 'output', file)} {os.path.join('demphire_coord_table', dir, 'output', file)}")

