import sys, os
sys.path.append("../")

import numpy as np
from vocab.vocab import SequenceVocabulary
'''
Vectorizing sequence
'''
class MIMICVectorizer:
    """
    Vectorizer to convert a sequence of ICD10 code to vectorized arrays
    """

    def __init__(self, diagnoses_vocab: SequenceVocabulary):
        self.diagnoses_vocab = diagnoses_vocab
        self.diagnoses_vocab_len = len(self.diagnoses_vocab)

    def vectorize(self, diagnoses: str):
        """
        Convert the diagnoses sequence to a list of indeces based on the
        vocabulary object. 
        This method also handles the padding of the "inner-list", i.e.,
        the number of diagnoses per visit. Padding the number of visits sequence
        is handled by the collate function in trainer.
        
        Args:
          diagnoses: String of diagnoses for a patient, each visit 
                     separated by ';', diagnoses per 
                     visit separated by space
        Returns:
          diag_for_patient_padded: ndarray of shape 
                                   [num_visits, max_num_diagnoses_per_visit] 
          patient_num_visits: number of visits for this patient.
        """
        visits = [visit for visit in diagnoses.split(";")] # split visits from sequence
        patient_number_of_visits = len(visits)
        item_per_visit = []
        max_items_per_visit_length = 0
        for visit in visits:
            items_per_visit_i = [
                self.diagnoses_vocab.lookup_token(token) for token in visit.split("<SEP>")
            ]
            items_per_visit_i_length = len(items_per_visit_i)
            if max_items_per_visit_length < items_per_visit_i_length:
                max_items_per_visit_length = items_per_visit_i_length
            item_per_visit.append(items_per_visit_i)

        item_per_visit_padded = np.zeros(
            (patient_number_of_visits, max_items_per_visit_length), dtype=np.int64
        )

        for i, visit in enumerate(item_per_visit):
            item_per_visit_padded[i, : len(visit)] = visit

        return item_per_visit_padded, patient_number_of_visits

    @classmethod
    def from_dataframe(cls, df, colname="X_seq"):
        diagnoses_vocab = SequenceVocabulary()
        for _, row in df.iterrows():
            diagnoses_vocab.add_tokens(row[f"{colname}"].replace(";", "<SEP>").split("<SEP>"))

        print(f"Corpus has {len(diagnoses_vocab)} unique tokens")
        return cls(diagnoses_vocab)

# Try vectorizing the preprocessed MIMIC data
#df = pd.read_csv(f'data/samples.csv')
#df.sequential_code = df.sequential_code.str.replace(" ", "") # remove all spaces

#vectorizer = MIMICVectorizer.from_dataframe(df, "sequential_code")

# Example
#vectorizer.vectorize("9_5723<SEP>9_78959<SEP>9_5715<SEP>9_07070<SEP>9_496<SEP>9_29680<SEP>9_30981<SEP>9_V1582<SEP>IP_9_A419;9_07071<SEP>9_78959<SEP>9_2875<SEP>9_2761<SEP>9_496<SEP>9_5715<SEP>9_V08<SEP>9_3051<SEP>IP_9_S51812A;9_07054<SEP>9_78959<SEP>9_V462<SEP>9_5715<SEP>9_2767<SEP>9_2761<SEP>9_496<SEP>9_V08<SEP>9_3051<SEP>9_78791<SEP>IP_9_F319;9_45829<SEP>9_07044<SEP>9_7994<SEP>9_2761<SEP>9_78959<SEP>9_2767<SEP>9_3051<SEP>9_V08<SEP>9_V4986<SEP>9_V462<SEP>9_496<SEP>9_29680<SEP>9_5715")
