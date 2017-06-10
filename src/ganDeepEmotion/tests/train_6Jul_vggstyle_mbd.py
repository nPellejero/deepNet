from train_dcgan_stl10_generic import dcgan_experiment

experiment = dcgan_experiment('src.model_32x_vggstyle_mbatch_disc')
experiment.run()
