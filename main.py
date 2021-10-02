
from src.models.tabular import tabularmodel
from src.dataloaders.tabular import tabularData
from src.predictions.analyze_preds import prediction_plotter
from src.utils.dataModifier import set_seed

import warnings
warnings.filterwarnings("ignore")

import logging

#logging.basicConfig(filename='testrun.log' , level = logging.INFO , format = '%(asctime)s:%(levelname)s:%(messages)s:')


# Create Data Module
dataloader.analyze()
dataloader.plot_continous_variables(subsample=False, subsample_size=0.5, plot_per_row=5)
dataloader.preprocess(show_before_after=True)
dataloader.setup()

dataloader = tabularData(data_path, target_column=target_column, ignore_col=ignore_col)
# print(dataloader.analyze())
dataloader.preprocess(show_before_after=True)
# dataloader.plot_continous_variables(subsample = False , subsample_size = 0.5 , plot_per_row = 5)

model = tabularmodel(architecture=architecture, problem=problem)
model.fit(dataloader)
predictions = model.predict()

plotter = prediction_plotter(predictions, dataloader)
plotter.compute_metrics()


@hydra.main(config_path="configs", config_name="tabular")
def tabular_hydra(cfg: DictConfig):
    pl.seed_everything(1234)

    # Log Metrics using Wandb
    wandb_logger = instantiate(cfg.wandb)
    wandb_logger.log_hyperparams(cfg)

    # Create Data Module
    dataloader = instantiate(cfg.dataloader)
    dataloader.analyze()
    dataloader.plot_continous_variables(subsample=False, subsample_size=0.5, plot_per_row=5)
    dataloader.preprocess(show_before_after=True)
    dataloader.setup()


    dataloader = tabularData(data_path, target_column=target_column, ignore_col=ignore_col)
    # print(dataloader.analyze())
    dataloader.preprocess(show_before_after=True)
    # dataloader.plot_continous_variables(subsample = False , subsample_size = 0.5 , plot_per_row = 5)

    model = tabularmodel(architecture=architecture, problem=problem)
    model.fit(dataloader)
    predictions = model.predict()

    plotter = prediction_plotter(predictions, dataloader)
    plotter.compute_metrics()



    # Create Model
    model = instantiate(cfg.model)

    # Training
    trainer = TabularTrainer(logger=[wandb_logger], **cfg.trainer )
    trainer.fit(model=model, datamodule=data_module)

    trainer.



@hydra.main(config_path="configs", config_name="imageClassification")
def cnn_hydra(cfg: DictConfig):
    pl.seed_everything(1234)

    set_seed(random_seed)

    # Log Metrics using Wandb
    wandb_logger = instantiate(cfg.wandb)
    wandb_logger.log_hyperparams(cfg)

    # Create Data Module
    dataloader = instantiate(cfg.data)
    dataloader.prepare_data()
    dataloader.setup()

    dataloader.sample()


    # Create Model
    model = instantiate(cfg.model)

    # Training
    trainer = pl.Trainer(logger=[wandb_logger], **cfg.trainer )
    trainer.fit(model=model, datamodule=data_module)

