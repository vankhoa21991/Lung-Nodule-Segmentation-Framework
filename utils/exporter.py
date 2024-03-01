import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from pathlib import Path

class Exporter:
	def __init__(self, output_dir):
		self.output_dir = Path(output_dir).parent

	def export2d(self, img, msk, pred, name, mode, dataset, model_name, loss_name):
		output_dir = Path(self.output_dir) / 'export' / f'{mode}_{dataset}_{model_name}_{loss_name}'
		os.makedirs(output_dir, exist_ok=True)
		if isinstance(img, torch.Tensor):
			img = img.cpu().detach().numpy().squeeze()
			msk = msk.cpu().detach().numpy().squeeze()
			pred = pred.cpu().detach().numpy().squeeze()

		fig, plots = plt.subplots(1, 3)
		plots[0].imshow(img, cmap='gray')
		plots[0].set_xlabel('image')
		plots[1].imshow(msk, cmap='gray')
		plots[1].set_xlabel('ground truth')
		plots[2].imshow(pred, cmap='gray')
		plots[2].set_xlabel('prediction')
		plt.savefig(f'{output_dir}/{name[0]}.png')
		plt.close()

	def export3d(self, img, msk, pred, name, mode, dataset, model_name, loss_name):
		output_dir = Path(self.output_dir) / 'export' / f'{mode}_{dataset}_{model_name}_{loss_name}'
		os.makedirs(output_dir, exist_ok=True)
		if isinstance(img, torch.Tensor):
			img = img.cpu().detach().numpy().squeeze()
			msk = msk.cpu().detach().numpy().squeeze()
			pred = pred.cpu().detach().numpy().squeeze()

		img = sitk.GetImageFromArray(img)
		msk = sitk.GetImageFromArray(msk)
		pred = sitk.GetImageFromArray(pred)

		sitk.WriteImage(img, f'{output_dir}/{name[0]}_img.nii')
		sitk.WriteImage(msk, f'{output_dir}/{name[0]}_msk.nii')
		sitk.WriteImage(pred, f'{output_dir}/{name[0]}_pred.nii')