"""Tokenize radiology report impressions and save as a list."""

from tqdm import tqdm


def get_impressions_from_pandas(df):
    """Get the report impressions from a pandas DataFrame and clean them."""
    imp = df["Report Impression"]
    imp = imp.str.strip()
    imp = imp.replace(r"\n", " ", regex=True)
    imp = imp.replace(r"\s+", " ", regex=True)
    imp = imp.str.strip()
    return imp


def tokenize(impressions, tokenizer, verbose=True):
    """Tokenize radiology report impressions and save as a list.

    Args:
        impressions: The reports to tokenize.
        tokenizer: The tokenizer to use.
        verbose: Defaults to True.

    Returns:
        The tokenized reports.
    """
    # raise Exception
    new_impressions = []
    if verbose:
        print("\nTokenizing report impressions. All reports are cut off at 512 tokens.")
    for i in tqdm(range(impressions.shape[0]), disable=not verbose):
        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        if tokenized_imp:  # not an empty report
            res = tokenizer.encode_plus(tokenized_imp)["input_ids"]
            if len(res) > 512:  # length exceeds maximum size
                # print("report length bigger than 512")
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:  # an empty report
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions
