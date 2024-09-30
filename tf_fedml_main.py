import logging
import fedml
from fedml import FedMLRunner
from tf_model_trainer import TfModelTrainerCLS
from tf_model_aggregator import TfServerAggregator
from tf_model import unet_3d, dice_coef, soft_dice_coef, dice_loss
from data_loader import load_data
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
if __name__ == "__main__":
    # install("tensorflow")
    # init FedML framework
    logging.error("client/server - started")
    args = fedml.init()
    logging.error("client/server - args loaded")

    # init device
    device = fedml.device.get_device(args)
    logging.error("client/server - device inited")
    
    # load data
    dataset = load_data(args)
    logging.error("client/server - data loaded")

    model = unet_3d((args.input_dim, args.input_dim, args.input_dim, args.input_channels))
    logging.error("client/server - model created")
    
    from tensorflow.keras.optimizers import Adam
    
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss= dice_loss,
        metrics=[dice_coef, soft_dice_coef]
    )
    logging.error("client/server - model compiled")
    
    # create model trainer
    model_trainer = TfModelTrainerCLS(model, dataset, args)
    logging.error("client/server - model trainer created")
    aggregator = TfServerAggregator(model, dataset, args)
    logging.error("client/server - aggregator created")

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer=model_trainer, server_aggregator=aggregator)
    fedml_runner.run()
    logging.error("client/server - fedml run is run")
