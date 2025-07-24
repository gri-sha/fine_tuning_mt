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
from util import generate_instruction_prompts, initialize_dfs, TEST_SPLIT

with open("translate/with_api/MistralAI/mistralAI_config.yml", "r") as f:
    config = yaml.safe_load(f)

_, df_test = initialize_dfs(test=TEST_SPLIT)
s0, p0, r0 = generate_instruction_prompts(df_test, shots=0)
s1, p1, r1 = generate_instruction_prompts(df_test, shots=1, fuzzy=True)

dataset_0shot = pd.DataFrame({"sources": s0, "references": r0, "prompts": p0})
dataset_1shot = pd.DataFrame({"sources": s1, "references": r1, "prompts": p1})

sources = s0 + s1
references = r0 + r1

# create the batch file
batch = []
for idx, row in dataset_0shot.iterrows():
    elem = {
        "custom_id": str(idx + 1),
        "body": {
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "messages": [
                {
                    "role": "system",
                    "content": config["system_prompt_0shot"],
                },
                {
                    "role": "user",
                    "content": row.prompts,
                },
            ],
        },
    }
    batch.append(elem)
last_idx = len(dataset_0shot)

for idx, row in dataset_1shot.iterrows():
    elem = {
        "custom_id": str(idx + last_idx + 1),
        "body": {
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "messages": [
                {
                    "role": "system",
                    "content": config["system_prompt_few_shot"],
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
        "file_name": config["batch_name"],
        "content": open(os.path.join(config["batch_dir"], config["batch_file"]), "rb")
    },
    purpose = "batch"
)

job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model=config['model'],
    endpoint=config['endpoint'],
)

# wait until the job is completed
while True:
    # Check if job is completed
    job = client.batch.jobs.get(job_id=job.id)
    
    if job.status in ['SUCCESS', 'FAILED', 'CANCELLED']:
        print(f"Job {job.id} finished with status: {job.status}")
        break
    
    # If not completed, wait before checking again
    print(f"Job {job.id} current status: {job.status}. Waiting...")
    time.sleep(config['check_interval'])

# save the results
if job.status == 'SUCCESS':
    if job.output_file:
        output_file_stream = client.files.download(file_id=job.output_file)
        with open(os.path.join(config["batch_dir"], config["batch_results_file"]), 'wb') as f:
            f.write(output_file_stream.read())
        print("Results downloaded successfully.")
    else:
        print("Job completed successfully but no output file was generated.")
        sys.exit()
else:
    print(f"Job did not complete successfully. Status: {job.status}")
    sys.exit()

data = []
with open(os.path.join(config["batch_dir"], config["batch_results_file"]), 'r') as f:
    for line in f:
        line = json.loads(line)
        data.append((int(line["custom_id"]), line['response']['body']['choices'][0]['message']['content']))

data.sort() # by default sorts by the first value
translations = [elem[1] for elem in data]

os.makedirs(config["translations_dir"], exist_ok=True)
translations_df = pd.DataFrame(
    {"sources": sources, "references": references, "translations": translations}
)
translations_df.to_csv(os.path.join(config["translations_dir"], config["translations_file"]), index=False)
