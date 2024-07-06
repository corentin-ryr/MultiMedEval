"""Module to define the dataset for unlabeled data."""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from multimedeval.chexbert import bert_tokenizer


class UnlabeledDataset(Dataset):
    """The dataset to contain report impressions without any labels."""

    def __init__(self, df, verbose=True):
        """Initialize the dataset object."""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        impressions = bert_tokenizer.get_impressions_from_pandas(df)
        self.encoded_imp = bert_tokenizer.tokenize(
            impressions, tokenizer, verbose=verbose
        )

    def __len__(self):
        """Compute the length of the dataset.

        Returns:
            Size of the dataframe
        """
        return len(self.encoded_imp)

    def __getitem__(self, idx):
        """Functionality to index into the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imp = self.encoded_imp[idx]
        imp = torch.LongTensor(imp)
        return {"imp": imp, "len": imp.shape[0], "idx": idx}
