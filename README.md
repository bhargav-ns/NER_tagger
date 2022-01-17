Primary script for submission is in `crf_ner_tagger-2.py`

The submission (predictions on test data) is in `test_data_predictions-2.txt`

`train_meta_files-2` consists of the validation set (`goldstandardfile.txt`) and its respective output in (`yoursystemoutput.txt`). You can run these files to coroborate our reported F1 score.

### Instructions to run program :

1. Run requirements.txt : pip install -r requirements.txt
2. Run ner_tagger.py : python crf_ner_tagger-2.py
   - This produces the outputs : test_data_predictions.txt, goldstandardfile.txt, yoursystemoutput.txt
   - Please ensure that S21-gene-train.txt, F21-gene-test.txt and ner_tagger.py are in the same directory.
