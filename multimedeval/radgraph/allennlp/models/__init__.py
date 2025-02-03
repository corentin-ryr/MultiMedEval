"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of `Model`.
"""

from multimedeval.radgraph.allennlp.models.model import Model
from multimedeval.radgraph.allennlp.models.archival import (
    archive_model,
    load_archive,
    Archive,
)
from multimedeval.radgraph.allennlp.models.simple_tagger import SimpleTagger
from multimedeval.radgraph.allennlp.models.basic_classifier import BasicClassifier
