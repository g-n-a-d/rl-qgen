import jsonlines
from rouge_score import rouge_scorer
import evaluate
from bert_score import score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eval_filename", type=str, default="./pred.jsonl", help="Evaluation filename formatted in jsonl")
args = parser.parse_args()

rouge = evaluate.load('rouge')
# bleu = evaluate.load('bleu')

target, pred = [], []

with jsonlines.open(args.eval_filename, mode="r") as f:
    for line in f:
        target.append(line["target"])
        pred.append(line["pred"])

rouge_score = rouge.compute(predictions=pred, references=target)
# bleu_score = 
bs_pre, bs_rec, bs_f1 = score(pred, target, lang="vi", verbose=True)
print(bs_pre, bs_rec, bs_f1)

print("#### Overall Mean Scores ####")
print("+++++++++++++++++")
print("Rouge")
print("Rouge1: {:.2f}".format(100*rouge_score["rouge1"]))
print("Rouge2: {:.2f}".format(100*rouge_score["rouge2"]))
print("RougeL: {:.2f}".format(100*rouge_score["rougeL"]))
# print("+++++++++++++++++")
# print("Bleu-4")
# print('Precision: {:.4f}'.format(100*sum(bleu_pre)/len(bleu_pre)))
# print('Recall: {:.4f}'.format(100*sum(bleu_rec)/len(bleu_pre)))
# print('F1 score: {:.4f}'.format(100*sum(bleu_f1)/len(bleu_f1)))
print("+++++++++++++++++")
print("BERTScore")
print("Precision: {:.2f}".format(100*bs_pre))
print("Recall: {:.2f}".format(100*bs_rec))
print("F1 score: {:.2f}".format(100*bs_f1))
print("+++++++++++++++++")