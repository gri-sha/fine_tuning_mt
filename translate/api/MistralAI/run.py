import os
import sys
import yaml
import json
import time
import pandas as pd
from mistralai import Mistral
from pprint import pprint
from dotenv import load_dotenv

# load the dataset is the same configuration as for finetuning
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from util import _generate_instruction_prompts, _initialize_dfs, create_test_dataset

with open("translate/with_api/MistralAI/mistralAI_config.yml", "r") as f:
    config = yaml.safe_load(f)

dataset = create_test_dataset(shots=config["shots"], fuzzy=config["fuzzy"])
df = dataset.to_pandas()

# create the batch file
batch = []
for idx, row in df.iterrows():
    elem = {
        "custom_id": str(idx + 1),
        "body": {
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "messages": [
                {
                    "role": "system",
                    "content": config["system_prompt"],
                },
                {
                    "role": "user",
                    "content": row.prompts,
                },
            ],
        },
    }
    batch.append(elem)


os.makedirs(config["batch_dir"], exist_ok=True)
with open(os.path.join(config["batch_dir"], config["batch_file"]), "w") as f:
    for elem in batch:
        f.write(json.dumps(elem) + "\n")

# create job
load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

batch_data = client.files.upload(
    file={
        "file_name": config["batch_file"],
        "content": open(os.path.join(config["batch_dir"], config["batch_file"]), "rb"),
    },
    purpose="batch",
)

job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model=config["model"],
    endpoint=config["endpoint"],
)

# wait until the job is completed
while True:
    # Check if job is completed
    job = client.batch.jobs.get(job_id=job.id)

    if job.status in ["SUCCESS", "FAILED", "CANCELLED"]:
        print(f"Job {job.id} finished with status: {job.status}")
        break

    # If not completed, wait before checking again
    print(f"Job {job.id} current status: {job.status}. Waiting...")
    time.sleep(config["check_interval"])

# save the results
if job.status == "SUCCESS":
    if job.output_file:
        output_file_stream = client.files.download(file_id=job.output_file)
        with open(
            os.path.join(config["batch_dir"], config["batch_results_file"]), "wb"
        ) as f:
            f.write(output_file_stream.read())
        print("Results downloaded successfully.")
    else:
        print("Job completed successfully but no output file was generated.")
        sys.exit()
else:
    print(f"Job did not complete successfully. Status: {job.status}")
    sys.exit()

data = []
with open(os.path.join(config["batch_dir"], config["batch_results_file"]), "r") as f:
    for line in f:
        line = json.loads(line)
        data.append(
            (
                int(line["custom_id"]),
                line["response"]["body"]["choices"][0]["message"]["content"],
            )
        )

data.sort()  # by default sorts by the first value
translations = [elem[1] for elem in data]

directory, _ = os.path.split(config["translations_path"])
os.makedirs(directory, exist_ok=True)

translations_df = pd.DataFrame(
    {
        "sources": dataset["sources"],
        "references": dataset["references"],
        "translations": translations,
    }
)

translations_df.to_csv(config["translations_path"], index=False)
