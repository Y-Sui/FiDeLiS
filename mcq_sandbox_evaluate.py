import os
import argparse
from src.qa_prediction.evaluate_results import eval_result

def main(args):
   output_dir = os.path.join(args.output_path, args.model_name)
   eval_result(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), cal_f1=True)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--sample", type=int, default=-1)
   parser.add_argument("--data_path", type=str, default="rmanluo")
   parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
   parser.add_argument("--split", type=str, default="test")
   parser.add_argument("--output_path", type=str, default="results")
   parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
   parser.add_argument("--n_beam", type=int, default=1)
   parser.add_argument("--whether_filtering", type=bool, default=False)
   args = parser.parse_args()
   main(args)