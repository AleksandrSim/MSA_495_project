from argparse import ArgumentParser
import yaml
from pytorch_lightning import Trainer
from helper_functions import *
from gan_model import *
from gan_module import *
from global_variables import *

parser = ArgumentParser()
parser.add_argument('--config', default='../config_files/config.yaml', help='Config .yaml file to use for training')

if __name__ == '__main__':

    # To read the data directory from the argument given
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)

    generate_dir_if_not_exists(config['user_path'] + gan_model_path)
    generate_dir_if_not_exists(config['user_path'] + gan_generated_images)

    model = AgingGAN(config)
    trainer = Trainer(max_epochs=config['epochs'], gpus=config['gpus'], auto_scale_batch_size='binsearch')
    trainer.fit(model)
