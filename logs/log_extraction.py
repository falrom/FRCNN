"""Tools that reads Tensorflow log files and converts them to excel."""

from tensorboard.backend.event_processing import event_accumulator
import csv
import os

key = 'PSNR_test'

log_dir_path = r'DRRNoverfit_B1U9C64/BasketballDrive_1920x1080_50_000to049_QP25/20181020145419'
log_file_paths = sorted([os.path.join(log_dir_path, p) for p in os.listdir(log_dir_path) if p.split('.')[-1] == 'ubun'])
csv_file_path = os.path.join(log_dir_path, key + '.csv')

print('From:', log_dir_path)
print('To  :', csv_file_path)
print('Key :', key)

if os.path.exists(csv_file_path):
    os.remove(csv_file_path)
with open(csv_file_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['step', key])
    for log_file_path in log_file_paths:
        print()
        print('From:', os.path.split(log_file_path)[-1])
        ea = event_accumulator.EventAccumulator(log_file_path)
        ea.Reload()
        print('Keys:', ea.scalars.Keys())
        if key not in ea.scalars.Keys():
            raise Exception('Key word not found: ' + key)
        items = ea.scalars.Items(key)
        print('Length:', len(items))
        for it in items:
            writer.writerow([it.step, it.value])
