# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/raw/tep_data.csv')
@click.argument('output_filepath', type=click.Path(), default='data/processed/tep_data.csv')
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    raw_data = pd.read_csv(input_filepath, header=None)

    processed = StandardScaler().fit_transform(raw_data)
    logger.info('Saving proccesed data')
    pd.DataFrame(data=processed).to_csv(output_filepath, index_label='Index')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
