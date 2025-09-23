import os
import sys
import yaml
import comet
import sacrebleu
import pandas as pd
from pathlib import Path
from bert_score import score as bert
from pprint import pprint

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util import parse_evaluation_arguments
args = parse_evaluation_arguments()

# Get all the translation .csv files from the given folder
tr_paths = []
for root, _, files in os.walk(args.tr_dir):
    for file in files:
        if file.lower().endswith(".csv"):
            path = os.path.join(root, file)
            tr_paths.append(path)

print("Found translation files:")
pprint(tr_paths)

# Load the evaluations to dataframe
eval_df = (
        pd.read_csv(args.eval_path)
        if os.path.exists(args.eval_path)
        else pd.DataFrame()
    )

# Get comet config
with open("evaluate/comet_config.yml", "r") as f:
    comet_config = yaml.safe_load(f)

if comet_config["use_comet"]:
    model_path = os.path.join(
                    comet_config["comet_dir"],
                    comet_config["comet_model_name"],
                    "checkpoints/model.ckpt",
                )
    if not os.path.exists(model_path):
        model_path = comet.download_model(
            model=comet_config["comet_model_name"],
            saving_directory=comet_config["comet_dir"],
        )
    model = comet.load_from_checkpoint(model_path)


# Calculate scores
for tr_path in tr_paths:
    try:
        # Get eval name 
        p = Path(tr_path)
        eval_name = f"{p.parent.name}_{p.stem}"

        # Extract translations
        df = pd.read_csv(tr_path)
        sources = df["sources"].tolist()
        references = df["references"].tolist()
        translations = df["translations"].tolist()

        # Calculate BLEU
        bleu = sacrebleu.corpus_bleu(
            translations, [references]
        )  # for spBLEU: tokenize='flores200'
        bleu = round(bleu.score, 2)
        print("BLEU:", bleu)

        # Calculate chrF++
        chrf = sacrebleu.corpus_chrf(
            translations, [references], word_order=2
        )  # for chrF++ word_order=2
        chrf = round(chrf.score, 2)
        print("chrF++:", chrf)

        # Calculate TER
        metric = sacrebleu.metrics.TER()
        ter = metric.corpus_score(translations, [references])
        ter = round(ter.score, 2)
        print("TER:", ter)

        # Calculate BERTScore
        P, R, F1 = bert(translations, references, lang="fr", rescale_with_baseline=True)
        bert_p = round(P.mean().item() * 100, 2)
        bert_r = round(R.mean().item() * 100, 2)
        bert_f1 = round(F1.mean().item() * 100, 2)
        print("BERT (P):", bert_p)
        print("BERT (R):", bert_r)
        print("BERT (F1):", bert_f1)

        # Calculate COMET
        if comet_config["use_comet"]:
            comet_df = pd.DataFrame({"src": sources, "mt": translations, "ref": references})
            comet_data = comet_df.to_dict("records")

            seg_scores, sys_score = model.predict(comet_data, batch_size=128, gpus=1).values()
            comet_score = round(sys_score * 100, 2)
            print("COMET:", comet_score)

            new_row = {
                "name": eval_name,
                "BLEU": bleu,
                "chrF++": chrf,
                "TER": ter,
                "BERT (P)": bert_p,
                "BERT (R)": bert_r,
                "BERT (F1)": bert_f1,
                "COMET": comet_score,
            }
        else:
            new_row = {
                "name": eval_name,
                "BLEU": bleu,
                "chrF++": chrf,
                "TER": ter,
                "BERT (P)": bert_p,
                "BERT (R)": bert_r,
                "BERT (F1)": bert_f1,
            }

        eval_df = pd.concat([eval_df, pd.DataFrame([new_row])], ignore_index=True)
    except Exception as e:
        print(f"Error processing {tr_path}:\n{e}")
        continue

eval_df.to_csv(args.eval_path, index=False)
