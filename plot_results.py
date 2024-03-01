from configs import config
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_dice(mode):
	result_path = Path(config.csv_path) / 'evaluate'
	result_paths = list(result_path.glob('*.csv'))

	results = defaultdict(list)
	nrows = 0
	for result_path in result_paths:
		df = pd.read_csv(result_path)
		if len(df) < 18:
			continue
		mod = str(result_path).split('/')[-1].split('_')[2]
		if mod != mode:
			continue
		model = str(result_path).split('/')[-1].split('_')[0]
		dataset = str(result_path).split('/')[-1].split('_')[1]
		dice_dice = df['dsc'][5]
		dice_bce = df['dsc'][11]
		dice_focal = df['dsc'][17]

		results[nrows] = {
			'mode': mode,
			'model': model,
			'dataset': dataset,
			'dice_dice': dice_dice,
			'dice_bce': dice_bce,
			'dice_focal': dice_focal
		}
		nrows += 1

	results = pd.DataFrame(results).T

	# make bar plot for each model of each loss
	fig, ax = plt.subplots(figsize=(15, 5))
	sns.lineplot(x='model', y='dice_dice', data=results, ax=ax)
	sns.lineplot(x='model', y='dice_bce', data=results, ax=ax)
	sns.lineplot(x='model', y='dice_focal', data=results, ax=ax)
	ax.set_ylabel('Dice')
	ax.set_xlabel('Model')
	ax.set_title('Dice for different models and losses')
	ax.yaxis.set_ticks(np.linspace(50, 100, 5).astype(int))
	plt.ylim(0, 100)

	plt.legend(['dice', 'bce', 'focal'])
	plt.grid()
	plt.savefig(Path(config.csv_path) / f'dice{mode}.png')
	plt.show()

if __name__=='__main__':
	plot_dice(config.mode)


