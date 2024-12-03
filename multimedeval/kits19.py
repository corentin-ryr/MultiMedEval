import os
import datasets
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

from multimedeval.task_families import Segmentation
from multimedeval.utils import clean_str, download_file, BatcherInput, clone_repository
from glob import glob

from pathlib import Path
from shutil import move
import sys
import time

import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
import requests
import numpy as np
import nibabel as nib


class KITS19(Segmentation):
    """KITS19 Image Classification task."""

    def __init__(self, **kwargs):
        """Initialize the KITS19 Image Classification task."""
        super().__init__(**kwargs)
        self.modality = "CT"
        self.task_name = "KITS19"
        self.num_classes = 3

    def setup(self):
        """Setup the KITS19 Segmentation task."""
        self.num_classes = 14
        self.seg_type = "multilabel"
        # Get the dataset from Kaggle
        self.path = self.engine.get_config()["kits19_dir"]

        if self.path is None:
            raise ValueError(
                "Skipping KITS19 because the cache directory is not set."
            )

        self._generate_dataset()

        config_contents = [[i, self._get_destination(i), self._get_destination(i, False)] 
                           for i in range(300)]
        config_df = pd.DataFrame(columns=["index","img_path","seg_path"])
        self.dataset = datasets.Dataset.from_pandas(config_df)

    def get_predicted_answer(self, answer: np.ndarray):
        """Convert the predicted mask to one-hot encoding.

        Args:
            answer: The predicted segmentation mask.

        Returns:
            The one-hot encoded segmentation mask.
        """
        answer = torch.LongTensor(answer)
        one_hot_answer = one_hot(answer, num_classes= self.num_classes).movedim(-1, 1)

        return one_hot_answer

    def get_correct_answer(self, sample):
        """Returns the ground truth mask for the sample.

        Args:
            sample: The sample to get the correct mask from.

        Returns:
            The one-hot encoded ground truth mask.
        """
        gt_mask = nib.load(sample["seg_path"])
        gt_mask_tr = torch.LongTensor(gt_mask.get_fdata())

        one_hot_gt = one_hot(gt_mask_tr, num_classes= self.num_classes).movedim(-1, 1)

        return one_hot_gt

    def format_question(self, sample, prompt=False):
        """Formats the question.

        Args:
            sample: The sample to format.
            prompt: Adds the answer to the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with the formatted prompt,
              images, and segmentation mask.
        """
        batcher_input = BatcherInput()

        question = "<img> Please segment the kidney and tumor in the CT images."

        batcher_input._add_text_prompt("user", question)
        if prompt:
            batcher_input._add_text_prompt('assistant', "This is the answers <seg>.")
            seg = nib.load(sample["seg_path"])
            batcher_input._add_segmentation_mask(seg)

        image = nib.load(sample["img_path"])
        batcher_input._add_images(image)

        return batcher_input


    def _generate_dataset(self):
        '''
            Generate datasets through provided scripts on Kaggle, Data size: about 20 GB.
        '''
        if os.path.exists(os.path.join(self.path,"data")):
            return

        #Step 1: Download the repo, incl. segmentation mask under data folder
        repository_url = "https://github.com/neheller/kits19"
        target_directory = self.path

        clone_repository(repository_url, target_directory)

        #Step 2: Download the image data from a different source.
        def cleanup(bar, msg):
            bar.close()
            if os.path.exists(temp_f):
                os.remove(temp_f)
            print(msg)
            sys.exit()

        os.makedirs(self.path, exist_ok=True)

        imaging_url = "https://kits19.sfo2.digitaloceanspaces.com/"
        imaging_name_tmplt = "master_{:05d}.nii.gz"
        temp_f = os.path.join(self.path, 'temp.tmp')

        left_to_download = []
        for i in range(300):
            if not os.path.exists(self._get_destination(i)):
                left_to_download = left_to_download + [i]

        print("{} cases to download...".format(len(left_to_download)))
        for i, cid in enumerate(left_to_download):
            print("Download {}/{}: ".format(
                i+1, len(left_to_download)
            ))
            destination = self._get_destination(cid)
            remote_name = imaging_name_tmplt.format(cid)
            uri = imaging_url + remote_name 

            chnksz = 1000
            tries = 0
            while True:
                try:
                    tries = tries + 1
                    response = requests.get(uri, stream=True)
                    break
                except Exception as e:
                    print("Failed to establish connection with server:\n")
                    print(str(e) + "\n")
                    if tries < 1000:
                        print("Retrying in 30s")
                        time.sleep(30)
                    else:
                        print("Max retries exceeded")
                        sys.exit()

            try:
                with open(temp_f, "wb") as f:
                    bar = tqdm(
                        unit="KB", 
                        desc="case_{:05d}".format(cid), 
                        total=int(
                            np.ceil(int(response.headers["content-length"])/chnksz)
                        )
                    )
                    for pkg in response.iter_content(chunk_size=chnksz):
                        f.write(pkg)
                        bar.update(int(len(pkg)/chnksz))

                    bar.close()
                move(str(temp_f), str(destination))
            except KeyboardInterrupt:
                cleanup(bar, "KeyboardInterrupt")
            except Exception as e:
                cleanup(bar, str(e))
    
    def _get_destination(self,index, isImage = True):
        """
            Get the full path for sample of given index.
            index: integer index of the samples.
            isImage: Default True, returns the path of image data.False for segmentation.
        """
        if isImage:
            destination = os.path.join(
                self.path, 'data',  f"case_{index:05d}", "imaging.nii.gz"
                )
        else:
            destination = os.path.join(
                self.path, 'data',  f"case_{index:05d}", "segmentation.nii.gz"
                )
        
        dest_parent = os.path.join(self.path, 'data',  f"case_{index:05d}")
        if not os.path.exists(dest_parent):
            os.makedirs(dest_parent, exist_ok=True)
        return destination
