"""
Custom PyTorch
`Module <https://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP `Model` s.
"""

from multimedeval.radgraph.allennlp.modules.attention import Attention
from multimedeval.radgraph.allennlp.modules.bimpm_matching import BiMpmMatching
from multimedeval.radgraph.allennlp.modules.conditional_random_field import (
    ConditionalRandomField,
)
from multimedeval.radgraph.allennlp.modules.elmo import Elmo
from multimedeval.radgraph.allennlp.modules.feedforward import FeedForward
from multimedeval.radgraph.allennlp.modules.gated_sum import GatedSum
from multimedeval.radgraph.allennlp.modules.highway import Highway
from multimedeval.radgraph.allennlp.modules.input_variational_dropout import (
    InputVariationalDropout,
)
from multimedeval.radgraph.allennlp.modules.layer_norm import LayerNorm
from multimedeval.radgraph.allennlp.modules.matrix_attention import MatrixAttention
from multimedeval.radgraph.allennlp.modules.maxout import Maxout
from multimedeval.radgraph.allennlp.modules.residual_with_layer_dropout import (
    ResidualWithLayerDropout,
)
from multimedeval.radgraph.allennlp.modules.scalar_mix import ScalarMix
from multimedeval.radgraph.allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from multimedeval.radgraph.allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from multimedeval.radgraph.allennlp.modules.text_field_embedders import (
    TextFieldEmbedder,
)
from multimedeval.radgraph.allennlp.modules.time_distributed import TimeDistributed
from multimedeval.radgraph.allennlp.modules.token_embedders import (
    TokenEmbedder,
    Embedding,
)
from multimedeval.radgraph.allennlp.modules.softmax_loss import SoftmaxLoss
