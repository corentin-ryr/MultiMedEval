import logging
import os
from utils import Benchmark

class MIMIC_CXR_classification(Benchmark):
    def __init__(self, task_path, seed=1111):
        logging.debug('***** Transfer task : SICK-Relatedness*****\n\n')
        self.seed = seed
        
        # Get the dataset

    def do_prepare(self, params, prepare):
        # Call the prepare function if necessary (it shouldn't)


        return prepare(params, [])

    def run(self, params, batcher):
        
        # Run the batcher for all data


        # Compute the scores

        return {'devpearson': devpr, 'pearson': pr, 'spearman': sr, 'mse': se,
                'yhat': yhat, 'ndev': len(devA), 'ntest': len(testA)}

    