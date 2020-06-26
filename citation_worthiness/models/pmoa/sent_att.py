from typing import Dict, Optional

import torch
import torch.nn.functional as F
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from allennlp.modules.attention.legacy_attention import Attention
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Auc
from overrides import overrides

from citation_worthiness.metrics.auc_pr import AucPR


@Model.register("sent_att")
class SentAtt(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sent_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 encoder_attention: Attention = DotProductAttention(normalize=True),
                 label_namespace: str = "labels",
                 using_extra_len_feature=True,
                 class_weight=[1.0, 1.0],
                 dropout: Optional[float] = None,
                 calculate_f1: bool = None,
                 calculate_auc: bool = None,
                 calculate_auc_pr: bool = None,
                 positive_label: int = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SentAtt, self).__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        self.num_tags = self.vocab.get_vocab_size()
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.sent_encoder = sent_encoder
        self.attention = encoder_attention
        self.using_extra_len_feature = using_extra_len_feature

        # self.attention_scale = math.sqrt(encoder.get_output_dim())

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.classifier_feedforward = classifier_feedforward
        if classifier_feedforward is not None:
            output_dim = classifier_feedforward.get_output_dim()

        self.metrics = {
            "accuracy": CategoricalAccuracy()
            }

        if isinstance(class_weight, list) and len(class_weight)>0:
            # [0.2419097587861097, 1.0]
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight))
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        self.positive_label = positive_label
        self.calculate_f1 = calculate_f1
        self.calculate_auc = calculate_auc
        self.calculate_auc_pr = calculate_auc_pr

        if calculate_f1:
            self._f1_metric = F1Measure(positive_label)

        if calculate_auc:
            self._auc = Auc(positive_label)
        if calculate_auc_pr:
            self._auc_pr = AucPR(positive_label)

        # check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
        #                        "text field embedding dim", "encoder input dim")

        if classifier_feedforward is not None:
            check_dimensions_match(sent_encoder.get_output_dim() + 2 if using_extra_len_feature else 0,
                                   classifier_feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")

        initializer(self)

        # def count_params(model):
        #     "count number trainable parameters in a pytorch model"
        #     total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
        #     return total_params
        # total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in self.parameters())
        # print("================================= total_params = {} ===============================".format(
        # total_params))

    @overrides
    def forward(self,  # type: ignore
                cur_sent: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                additional_features: torch.Tensor = None,
                **kwargs
                ) -> Dict[str, torch.Tensor]:

        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        section_name : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_tensor()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.

        cur_sent : Dict[str, Variable], required
            The output of ``TextField.as_array()``.

        label : Variable
            A variable representing the label for each instance in the batch.

        additional_features :  Variable
            A variable representing the additional features for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        def build_encoder_layer(field, encoder):
            mask = util.get_text_field_mask(field)
            embedded_text_input = self.text_field_embedder(field)
            if self.dropout:
                embedded_text_input = self.dropout(embedded_text_input)
            encoded_text = encoder(embedded_text_input, mask)
            if self.dropout:
                encoded_text = self.dropout(encoded_text)
            # (batch_size, num_directions * hidden_size)
            try:
                final_hidden_states = util.get_final_encoder_states(encoded_text, mask, encoder.is_bidirectional())
            except:
                print(field)

            # Add attention here
            attention_weights = self.attention(final_hidden_states, encoded_text, mask).unsqueeze(-1)

            # rnn_output (batch_size, seq_len, num_directions * hidden_size)
            # ==> (batch_size, num_directions * hidden_size, seq_len)
            rnn_output_re_order = encoded_text.permute(0, 2, 1)
            attention_output = torch.bmm(rnn_output_re_order, attention_weights).squeeze(-1)
            return attention_output

        final_states_cur_sent = build_encoder_layer(cur_sent, self.sent_encoder)

        if self.using_extra_len_feature:
            embeded_features = torch.cat((final_states_cur_sent, additional_features), dim=-1)
        else:
            embeded_features = final_states_cur_sent
        logits = self.classifier_feedforward(embeded_features)

        output_dict = {
            "logits": logits,
            "golden_label": label
            }

        if label is not None:
            loss = self.loss(logits, label)
            # using loss.item() to release the graph
            output_dict["loss"] = loss
            for metric in self.metrics.values():
                metric(logits, label)
            if self.calculate_f1:
                self._f1_metric(logits, label)

            class_probabilities = F.softmax(logits, dim=-1)
            output_dict['class_probabilities'] = class_probabilities
            positive_class_prob = class_probabilities[:, 1].detach()
            # label_prob, label_index = torch.max(class_probabilities, -1)
            # argmax_indices = numpy.argmax(class_probabilities, axis=-1)
            if self.calculate_auc:
                self._auc(positive_class_prob, label)
            if self.calculate_auc_pr:
                self._auc_pr(positive_class_prob, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in
            self.metrics.items()
            }

        if self.calculate_f1:
            p_r_f1 = self._f1_metric.get_metric(reset=reset)
            precision, recall, f1_measure = p_r_f1
            f1_dict = {'precision': precision, 'recall': recall, "f1": f1_measure}
            metrics_to_return.update(f1_dict)

        if self.calculate_auc:
            metrics_to_return["auc"] = self._auc.get_metric(reset=reset)
        if self.calculate_auc_pr:
            metrics_to_return["auc_pr"] = self._auc_pr.get_metric(reset=reset)

        return metrics_to_return
