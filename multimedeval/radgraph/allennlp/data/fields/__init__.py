"""
A :class:`~allennlp.data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from multimedeval.radgraph.allennlp.data.fields.field import Field
from multimedeval.radgraph.allennlp.data.fields.adjacency_field import AdjacencyField
from multimedeval.radgraph.allennlp.data.fields.array_field import ArrayField
from multimedeval.radgraph.allennlp.data.fields.flag_field import FlagField
from multimedeval.radgraph.allennlp.data.fields.index_field import IndexField
from multimedeval.radgraph.allennlp.data.fields.label_field import LabelField
from multimedeval.radgraph.allennlp.data.fields.list_field import ListField
from multimedeval.radgraph.allennlp.data.fields.metadata_field import MetadataField
from multimedeval.radgraph.allennlp.data.fields.multilabel_field import MultiLabelField
from multimedeval.radgraph.allennlp.data.fields.namespace_swapping_field import (
    NamespaceSwappingField,
)
from multimedeval.radgraph.allennlp.data.fields.sequence_field import SequenceField
from multimedeval.radgraph.allennlp.data.fields.sequence_label_field import (
    SequenceLabelField,
)
from multimedeval.radgraph.allennlp.data.fields.span_field import SpanField
from multimedeval.radgraph.allennlp.data.fields.text_field import TextField
