"""
This module contains classes representing regularization schemes
as well as a class for applying regularization to parameters.
"""

from multimedeval.radgraph.allennlp.nn.regularizers.regularizer import Regularizer
from multimedeval.radgraph.allennlp.nn.regularizers.regularizers import L1Regularizer
from multimedeval.radgraph.allennlp.nn.regularizers.regularizers import L2Regularizer
from multimedeval.radgraph.allennlp.nn.regularizers.regularizer_applicator import (
    RegularizerApplicator,
)
