import os

files = [
    'jour_pluie.pdf','mean_pluie_jour.pdf','nr_pluie_jour.pdf','nr_pluie_horaire.pdf',
    'jour_pluie_diff.pdf','mean_pluie_jour_diff.pdf','nr_pluie_jour_diff.pdf','nr_pluie_horaire_diff.pdf',
    'jour_pluie_rdiff.pdf','mean_pluie_jour_rdiff.pdf','nr_pluie_jour_rdiff.pdf','nr_pluie_horaire_rdiff.pdf'
]

total = 0
for f in files:
    sz = os.path.getsize(f'figures/{f}')
    total += sz
    print(f'{f}: {sz/1024:.0f} KB')
print(f'\nTotal: {total/1024/1024:.2f} MB')
