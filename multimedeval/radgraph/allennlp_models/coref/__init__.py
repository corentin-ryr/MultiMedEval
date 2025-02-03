"""
Coreference resolution is defined as follows: given a document, find and cluster entity mentions.
"""

from multimedeval.radgraph.allennlp_models.coref.dataset_readers.conll import (
    ConllCorefReader,
)
from multimedeval.radgraph.allennlp_models.coref.dataset_readers.preco import (
    PrecoReader,
)
from multimedeval.radgraph.allennlp_models.coref.dataset_readers.winobias import (
    WinobiasReader,
)
from multimedeval.radgraph.allennlp_models.coref.models.coref import CoreferenceResolver
from multimedeval.radgraph.allennlp_models.coref.predictors.coref import CorefPredictor
