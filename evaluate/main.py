import os
import yaml
import comet
import sacrebleu
import pandas as pd


with open("evaluate/eval_config.yml", "r") as f:
    eval_config = yaml.safe_load(f)

df = pd.read_csv(eval_config["translations_path"])
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

# Calculate COMET
df = pd.DataFrame({"src": sources, "mt": translations, "ref": references})
data = df.to_dict("records")

model_path = os.path.join(
    eval_config["comet_dir"], eval_config["comet_model_name"], "checkpoints/model.ckpt"
)
if not os.path.exists(model_path):
    model_path = comet.download_model(
        model=eval_config["comet_model_name"], saving_directory=eval_config["comet_dir"]
    )
model = comet.load_from_checkpoint(model_path)

seg_scores, sys_score = model.predict(data, batch_size=128, gpus=1).values()
comet = round(sys_score * 100, 2)
print("COMET:", comet)


df = (
    pd.read_csv(eval_config["evaluations_path"])
    if os.path.exists(eval_config["evaluations_path"])
    else pd.DataFrame()
)

new_row = {
    "model_name": eval_config["eval_name"],
    "BLEU": bleu,
    "chrF++": chrf,
    "TER": ter,
    "COMET": comet,
}

df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
df.to_csv(eval_config["evaluations_path"], index=False)
