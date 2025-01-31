"""Label reports using the BERT model."""

from collections import OrderedDict

import torch
from torch import nn
from tqdm import tqdm

from multimedeval.chexbert.constants import BATCH_SIZE, CONDITIONS, NUM_WORKERS, PAD_IDX
from multimedeval.chexbert.datasets.unlabeled_dataset import UnlabeledDataset
from multimedeval.chexbert.models.bert_encoder import BertEncoder
from multimedeval.chexbert.utils import generate_attention_masks


def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len where\
        the reports have no associated labels.

    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of
                                 each sequence in batch
    """
    tensor_list = [s["imp"] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=PAD_IDX
    )
    len_list = [s["len"] for s in sample_list]
    idx_list = [s["idx"] for s in sample_list]
    batch = {"imp": batched_imp, "len": len_list, "idx": idx_list}
    return batch


def load_unlabeled_data(
    df,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,  # pylint: disable=unused-argument
    shuffle=False,
):
    """Create UnlabeledDataset object for the input reports.

    @param df (string): dataframe containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not

    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(df, verbose=False)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return loader


class _label:
    def __init__(
        self,
        checkpoint_path,
        verbose=False,
        deepspeed=False,  # pylint: disable=unused-argument
    ) -> None:  # pylint: disable=unused-argument
        model = BertEncoder(logits=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 0:  # works even if only 1 GPU available
            if verbose:
                print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)  # type: ignore  # to utilize multiple GPU's
            model = model.to(device)
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )  # TODO check if it works
        else:
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu"), weights_only=True
            )
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model_state_dict"].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)

        model.eval()
        self.model = model
        self.verbose = verbose
        self.device = device

    def __call__(self, df):
        ld = load_unlabeled_data(df)

        y_pred = [[] for _ in range(len(CONDITIONS))]

        if self.verbose:
            print(
                "\nBegin report impression labeling. "
                "The progress bar counts the # of batches completed:"
            )
            print(f"The batch size is {BATCH_SIZE}")
        with torch.no_grad():
            for data in tqdm(ld, disable=not self.verbose):
                batch = data["imp"]  # (batch_size, max_len)
                batch = batch.to(self.device)
                src_len = data["len"]
                attn_mask = generate_attention_masks(batch, src_len, self.device)

                out = self.model(batch, attn_mask)

                for j, out_j in enumerate(out):
                    curr_y_pred = out_j.argmax(dim=1)  # shape is (batch_size)
                    y_pred[j].append(curr_y_pred)

            for pred_j in y_pred:
                pred_j = torch.cat(pred_j, dim=0)

        y_pred = [t.tolist() for t in y_pred]
        return y_pred


class _encode:

    def __init__(self, checkpoint_path, verbose=False, deepspeed=False) -> None:
        model = BertEncoder(False)
        if deepspeed:
            with deepspeed.zero.GatheredParameters(
                model.parameters(),
            ):
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 0:  # works even if only 1 GPU available
                if verbose:
                    print("Using", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(model)  # type: ignore  # to utilize multiple GPU's
                model = model.to(device)
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                model.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )  # TODO check if it works
            else:
                checkpoint = torch.load(
                    checkpoint_path, map_location=torch.device("cpu"), weights_only=True
                )
                new_state_dict = OrderedDict()
                for k, v in checkpoint["model_state_dict"].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=False)

        model.eval()
        self.model = model
        self.verbose = verbose
        self.device = device

    def __call__(self, df):
        ld = load_unlabeled_data(df)

        rep = []

        if self.verbose:
            print(
                "\nBegin report impression labeling. The progress bar "
                "counts the # of batches completed:"
            )
            print(f"The batch size is {BATCH_SIZE}")
        with torch.no_grad():
            for data in tqdm(ld, disable=not self.verbose):
                batch = data["imp"]  # (batch_size, max_len)
                batch = batch.to(self.device)
                src_len = data["len"]
                attn_mask = generate_attention_masks(batch, src_len, self.device)

                out = self.model(batch, attn_mask)
                for _, out_j in enumerate(out):
                    rep.append(out_j.to("cpu"))

        return torch.stack(rep)


# def save_preds(y_pred, csv_path, out_path):
#     """Save predictions as out_path/labeled_reports.csv
#     @param y_pred (List[List[int]]): list of predictions for each report
#     @param csv_path (string): path to csv containing reports
#     @param out_path (string): path to output directory
#     """
#     y_pred = np.array(y_pred)
#     y_pred = y_pred.T

#     df = pd.DataFrame(y_pred, columns=CONDITIONS)
#     reports = pd.read_csv(csv_path)["Report Impression"]

#     df["Report Impression"] = reports.tolist()
#     new_cols = ["Report Impression"] + CONDITIONS
#     df = df[new_cols]

#     df.replace(0, np.nan, inplace=True)  # blank class is NaN
#     df.replace(3, -1, inplace=True)  # uncertain class is -1
#     df.replace(2, 0, inplace=True)  # negative class is 0

#     df.to_csv(os.path.join(out_path, "labeled_reports.csv"), index=False)
