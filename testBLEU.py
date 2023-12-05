from torchmetrics.text import BLEUScore
from torchtext.data.metrics import bleu_score

n = 1

preds = ['the cat is on the mat']
targets = [['there is a cat on the mat', 'a cat is on the mat']]
bleu = BLEUScore(n_gram=n)
print(bleu(preds, targets))

preds = [pred.split() for pred in preds]
targets = [[sentence.split() for sentence in target] for target in targets]
print(preds)
print(targets)

print(bleu_score(preds, targets, max_n=n, weights=[1/n for _ in range(n)]))