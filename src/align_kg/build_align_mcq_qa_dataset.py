import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import os
import json
from datasets import load_dataset
import multiprocessing as mp
import utils
from tqdm import tqdm
from functools import partial

SEP = '<SEP>'
BOP = '<PATH>'
EOP = '</PATH>'


def build_data(args):
    '''
    Extract the paths between question and answer entities from the dataset.
    '''

    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d)

    print("Save results to: ", output_dir)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    # Load dataset
    if args.sample != -1:
        dataset = load_dataset(input_file, split=args.split).select(range(args.sample))
    else:
        dataset = load_dataset(input_file, split=args.split)
    with open(os.path.join(output_dir, args.save_name), 'w') as fout:
        with mp.Pool(args.n) as pool:

            for res in tqdm(
                pool.imap_unordered(
                    partial(process_data, remove_duplicate=args.remove_duplicate),
                    dataset,
                ),
                total=len(dataset),
            ):
                for r in res:
                    fout.write(json.dumps(r) + '\n')


def process_data(data, remove_duplicate=False):
    results = []
    question = data['question']
    graph = utils.build_graph(data['graph'])
    shortest_paths = utils.get_truth_paths(
        data['q_entity'], data['a_entity'], graph
    )  # [[(h, r, t), ...], ...]

    if remove_duplicate:
        shortest_paths = list(set(map(tuple, shortest_paths)))

    history = ""
    starting_entity = ""
    while shortest_paths:
        shortest_path = shortest_paths.pop(0)
        starting_flag = True
        for path in shortest_path:
            if starting_flag:
                h, r, t = path
                starting_entity = h
                history = utils.rule_to_string([h], sep_token=SEP, bop=BOP, eop=EOP)
                starting_flag = False
            else:
                history = history.replace("</PATH>", "")
                history = history + "<SEP>" + str(r) + "<SEP>" + str(t) + "</PATH>"
                h, r, t = path
            _, path_candidates, neighbors = (
                utils.get_entity_edges_with_neighbors_single(h, graph)
            )
            options = []
            for j, p in enumerate(path_candidates):
                options.append(
                    utils.rule_to_string(
                        [h, p, neighbors[j]], sep_token=SEP, bop=BOP, eop=EOP
                    )
                )
            options = "\n".join(options)
            results.append(
                {
                    "question": question,
                    "starting_entity": starting_entity,
                    "options": options,
                    "history": history,
                    "answer": path,
                }
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )  # this link is from the root of the dataset (https://huggingface.co/datasets/rmanluo/RoG-webqsp/viewer/default/test?p=15)
    parser.add_argument('--d', '-d', type=str, default='webqsp')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument("--output_path", type=str, default="/data/shared/yuansui/rog/AlignData")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument('--n', '-n', type=int, default=1)
    parser.add_argument('--remove_duplicate', action='store_true')
    parser.add_argument('--sample', type=int, default=-1)
    args = parser.parse_args()

    if args.save_name == "":
        if args.sample != -1:
            args.save_name = (
                args.d + "_mcq_" + args.split + "_sample" + str(args.sample) + ".jsonl"
            )
        else:
            args.save_name = args.d + "_mcq_" + args.split + ".jsonl"

    build_data(args)
