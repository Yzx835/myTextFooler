from myTextFooler.textfooler import TextFooler
import dataloader
from attack_classification import NLI_infer_BERT

texts, labels = dataloader.read_corpus('./data/ag')
# print(texts)
data = list(zip(texts, labels))

model_path = './bert_ag'
model = NLI_infer_BERT(model_path, nclasses=2, max_seq_length=128)
predictor = model.text_pred

textfooler = TextFooler(model=predictor, device='gpu0', IsTargeted=False, USE_model_path='./universal-sentence-encoder-large_3')


adv_xs = textfooler.generate(texts, labels)
print(adv_xs)
print(len(adv_xs))