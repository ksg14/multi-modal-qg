import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_stat (stats, split, key, save_path):
	plt.plot(stats [split] [key])
	plt.savefig(save_path / f'{split}_{key}.png')
	# plt.show()
	return

def best_epoch_stats (stats):
	best_train_epoch = np.argmin (stats ['train'] ['loss'])
	best_val_epoch = np.argmax (stats ['val'] ['bleu'])

	print (f"Best train epochs - {best_train_epoch} loss - {stats ['train'] ['loss'] [best_train_epoch]}")
	print (f"Best val epochs - {best_val_epoch} bleu - {stats ['val'] ['bleu'] [best_train_epoch]}")
	return


if __name__ == '__main__' :
	results_path = Path ('results/exp-1')
	stats_file = results_path / 'stats.json'
	
	with open (stats_file, 'r') as file_io:
		stats = json.load (file_io)

	best_epoch_stats (stats)
	plot_stat (stats, 'train', 'loss', results_path)
	plot_stat (stats, 'val', 'bleu', results_path)
