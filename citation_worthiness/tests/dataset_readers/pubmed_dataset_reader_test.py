from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from citation_worthiness.dataset_readers import CiteSentIdentDatasetReader
from pathlib import Path

class TestPubmedDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        reader = CiteSentIdentDatasetReader()
        instances = ensure_list(reader.read((Path(__file__) / '../../fixtures/pubmed_samples.jsonl').resolve()))
        instance0 = {"text": ["Annotation", "was", "performed", "using", "the", "IGS",
                              "Prokaryotic", "Annotation", "Engine", "."],
                     "label": 1}

        instance1 = {"text": ["Human", "MEST", "is", "endowed", "with", "two", "promoters",
                              "that", "use", "alternative", "first", "exons", "."],
                     "label": 0}

        assert len(instances) == 100

        fields = instances[0].fields
        assert [t.text for t in fields["text"].tokens] == instance0["text"]
        assert fields["label"].label == instance0["label"]

        fields = instances[1].fields
        assert [t.text for t in fields["text"].tokens] == instance1["text"]
        assert fields["label"].label == instance1["label"]