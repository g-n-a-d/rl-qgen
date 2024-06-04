import jsonlines
from rouge_score import rouge_scorer
import evaluate
from bert_score import score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eval_filename", type=str, default="./pred.jsonl", help="Evaluation filename formatted in jsonl")
args = parser.parse_args()

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

target, pred = [], []

with jsonlines.open(args.eval_filename, mode="r") as f:
    for line in f:
        target.append(line["target"])
        pred.append(line["pred"])

print("Calculating Rouge score...")
rouge_score = rouge.compute(predictions=pred, references=target)
print("Calculating BLEU score...")
bleu_score = bleu.compute(predictions=pred, references=target)
print("Calculating BERTScore...")
bs_pre, bs_rec, bs_f1 = score(pred, target, lang="vi", verbose=True)

print("#### Overall Mean Scores ####")
print("+++++++++++++++++")
print("Rouge score")
print("Rouge1: {:.2f}".format(100*rouge_score["rouge1"]))
print("Rouge2: {:.2f}".format(100*rouge_score["rouge2"]))
print("RougeL: {:.2f}".format(100*rouge_score["rougeL"]))
print("+++++++++++++++++")
print("BLEU score")
print("1-grams: {:.2f}".format(100*bleu_score["precisions"][0]))
print("2-grams: {:.2f}".format(100*bleu_score["precisions"][1]))
print("3-grams: {:.2f}".format(100*bleu_score["precisions"][2]))
print("4-grams: {:.2f}".format(100*bleu_score["precisions"][3]))
print("BLEU: {:.2f}".format(100*bleu_score["bleu"]))
print("+++++++++++++++++")
print("BERTScore")
print("Precision: {:.2f}".format(100*bs_pre.mean()))
print("Recall: {:.2f}".format(100*bs_rec.mean()))
print("F1 score: {:.2f}".format(100*bs_f1.mean()))
print("+++++++++++++++++")