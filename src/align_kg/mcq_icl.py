import argparse
from sklearn.metrics import confusion_matrix
import re
import numpy as np
import pathlib
import concurrent.futures
import json
import os
from openai import OpenAI


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='cora')
parser.add_argument('--gpt_response_folder', type=str, default='gpt_responses')
parser.add_argument('--overwrite_responses', type=bool, default=False)
args = parser.parse_args()
print(args)


def query_api(fpath, prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    print(type(response))
    print("PROMPT: ", prompt)
    print("=" * 20)
    print("RECEIVED RESPONSE: ", response)
    outputs_file = open(fpath, "w")
    outputs_file.write(json.dumps(response))


def load_answers_from_folder(folder, idx_to_load):
    answers = []
    for i in idx_to_load:
        try:
            response = json.load(open(f'{folder}/{i}.json', 'r'))
            answers.append(response['choices'][0]['message']['content'])
        except:
            print(f'File missing: {folder}/{i}.json')

    return answers


def run():
    classes = [
        'Case Based',
        'Genetic Algorithms',
        'Neural Networks',
        'Probabilistic Methods',
        'Reinforcement Learning',
        'Rule Learning',
        'Theory',
    ]
    class_str = ', '.join(classes)
    class_map = {x.lower(): i for i, x in enumerate(classes)}

    papers = [
        'Deep learning based image classification',
        'Training agents to solve the game of Go',
        'Proof that P is not equal to NP',
    ]
    labels = [2, 4, 6]  # ground truth labels for evaluation
    idx_to_query = range(
        len(papers)
    )  # sometimes we only want to query the API for a limited subset of papers

    prompts = []
    for idx in idx_to_query:
        prompts.append(
            f'Paper: {papers[idx]}\nQuestion: Which of the following categories does this paper belong to: {class_str}?\n\nAnswer: '
        )
    prompts = np.array(prompts)
    print("FIRST PROMPT")
    print(prompts[0])

    fpath_prefix = f'{args.gpt_response_folder}/{args.dataset_name}'
    pathlib.Path(fpath_prefix).mkdir(parents=True, exist_ok=True)
    fpaths = [f'{fpath_prefix}/{idx}.json' for idx in idx_to_query]
    if not args.overwrite_responses:
        fpaths = [
            f for f in fpaths if (not os.path.exists(f) or os.stat(f).st_size == 0)
        ]

    print(f'Running {len(fpaths)} API queries')

    for fpath, prompt in zip(fpaths, prompts):
        query_api(fpath, prompt)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #     res = executor.map(query_api, fpaths, prompts)

    answers = load_answers_from_folder(fpath_prefix, idx_to_query)


if __name__ == "__main__":
    run()
