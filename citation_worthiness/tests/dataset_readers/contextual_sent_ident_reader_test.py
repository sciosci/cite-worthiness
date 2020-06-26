import numpy as np
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from citation_worthiness.dataset_readers import ContextualCiteSentIdentDatasetReader

from pathlib import Path
class TestPubmedDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = ContextualCiteSentIdentDatasetReader()
        instances = ensure_list(reader.read((Path(__file__) / '../../fixtures/citation_sent_contextual.jsonl').resolve()))
        assert len(instances) == 2

        fields = instances[0].fields
        assert [t.text for t in fields["top_sec_name"].tokens] == ["abstract"]
        assert [t.text for t in fields["cur_sent"].tokens] == ["this", "is", "cur", "sentence"]
        additional_features = fields["additional_features"].array
        target = np.array([0, 8, 88, 16, 66, 1, 5, 55])

        assert all([additional_features[i] == target[i] for i in range(8)])
