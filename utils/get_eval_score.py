import argparse
import jsonlines
from rouge_score import rouge_scorer

parser = argparse.ArgumentParser()
parser.add_argument("--eval_filename", type=str, default="eval.jsonl", help="Evaluation filename formatted in jsonl")
args = parser.parse_args()

with jsonlines.open(args.eval_filename, mode="r") as f:
    writer = jsonlines.open(args.writer, "a")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL_pre, rougeL_rec, rougeL_f1 = [], [], []
    for line in f:
        score = scorer.score(line["question"], line["generated_question"])
        rougeL_pre.append(score["rougeL"].precision)
        rougeL_rec.append(score["rougeL"].recall)
        rougeL_f1.append(score["rougeL"].fmeasure)

print("#### Overall Mean Rouge-L Scores ####")
print('Precision: {:.4f}'.format(sum(rougeL_pre)/len(rougeL_pre)))
print('Recall: {:.4f}'.format(sum(rougeL_rec)/len(rougeL_pre)))
print('F1 score: {:.4f}'.format(sum(rougeL_f1)/len(rougeL_f1)))