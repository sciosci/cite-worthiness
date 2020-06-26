from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.nn.util import get_final_encoder_states
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.modules.similarity_functions import DotProductSimilarity

import math


@Model.register("att_bilstm_spd")
class AttBiLSTMScaledDotProduct(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 label_namespace: str = "labels",
                 dropout: Optional[float] = None,
                 calculate_f1: bool = None,
                 positive_label: int = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),

                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AttBiLSTMScaledDotProduct, self).__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder

        self.attention_scale = math.sqrt(encoder.get_output_dim())

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.classifier_feedforward = classifier_feedforward
        if classifier_feedforward is not None:
            output_dim = classifier_feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }

        self.loss = torch.nn.CrossEntropyLoss()

        self.positive_label = positive_label
        self.calculate_f1 = calculate_f1
        if calculate_f1:
            self._f1_metric = F1Measure(positive_label)

        # check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
        #                        "text field embedding dim", "encoder input dim")

        if classifier_feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), classifier_feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                # abstract: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                **kwargs
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        abstract : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        mask = util.get_text_field_mask(text)
        embedded_text_input = self.text_field_embedder(text)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        # Attention
        # (batch_size, num_directions * hidden_size)
        final_hidden_states = get_final_encoder_states(encoded_text, mask,
                                                       self.encoder.is_bidirectional())

        # (batch_size, num_directions * hidden_size, 1)
        f_unsqz = final_hidden_states.unsqueeze(-1)

        attention_scores = torch.bmm(encoded_text, f_unsqz)

        attention_scores_scaled = attention_scores / math.sqrt(attention_scores.size(-1))

        soft_attention_weights = F.softmax(attention_scores_scaled, 1)

        # rnn_output (batch_size, seq_len, num_directions * hidden_size)
        # ==> (batch_size, num_directions * hidden_size, seq_len)
        rnn_output_re_order = encoded_text.permute(0, 2, 1)

        # (batch_size, num_directions * hidden_size, seq_len) bmm (batch_size, seq_len, 1)
        # ==> (batch_size, num_directions * hidden_size, 1)
        attention_output = torch.bmm(rnn_output_re_order, soft_attention_weights).squeeze(-1)

        logits = self.classifier_feedforward(attention_output)
        output_dict = {
            "logits": logits,
            "mask": mask
        }

        if label is not None:
            loss = self.loss(logits, label)
            output_dict["loss"] = loss
            for metric in self.metrics.values():
                metric(logits, label)
            if self.calculate_f1:
                self._f1_metric(logits, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for metric_name, metric in
                             self.metrics.items()}

        if self.calculate_f1:
            p_r_f1 = self._f1_metric.get_metric(reset=reset)
            precision, recall, f1_measure = p_r_f1
            f1_dict = {'precision': precision, 'recall': recall, "f1": f1_measure}
            metrics_to_return.update(f1_dict)

        return metrics_to_return
