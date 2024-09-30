from path_rag import LLM_Backbone
import argparse
import multiprocessing as mp
import json
import os

with open("config.json", "r") as f:
    config = json.load(f)
    
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

x = ['hc4f82n', 'g.1s061n2yq', 'g.1hhc4q59w', 'g.1245_9g_3', 'g.1hhc40j_c', 'g.11b60gm1hy', 'm.0k5nssn', 'g.1hhc481r2', 'Donald Sangster', 'g.1hhc4fzvq', 'g.1hhc3yv49', 'm.0k5nt8g', 'g.1hhc3sb4d', 'g.1hhc4kjwc', 'm.0k7lljb', 'g.1hhc4q27s', 'g.1245_v000', 'm.0nfbzkp', 'm.0nfbzky', 'The Blue Lagoon', 'g.1hhc52mlh', 'g.11b60rqkq5', 'm.0nf75vj', 'g.1hhc4zjkq', 'g.1hhc4j594', 'g.1hhc3yf7_', 'Under radar', 'g.1hhc44zxg', 'g.1hhc4t10z', 'm.0nfbznz', 'g.1hhc3g__r', 'g.1245_kbtt', 'g.1hhc4tjx_', 'g.1hhc5310t', 'g.1hhc4sh88', 'g.1hhc3qx37', 'm.0nf5r22', 'g.1hhc3kp4p', 'g.1hhc3xpkz', 'm.0nf4wrt', 'g.1245_22xq', 'm.0nf75wm', 'g.1hhc4bpvb', 'g.1245zpmc7', 'g.12tb6gndj', 'g.1245zs0nr', 'g.12460846s', 'g.11b60tql_j', 'g.1hhc4n0gt', 'g.1hhc3jhxn', 'Hurricane Michelle', 'm.0nfb4qf', 'm.0nf98k1', 'g.1hhc4kjqh', 'm.0nfb4mr', 'g.1hhc3qbrl', 'Deon Hemmings', 'm.0nfbzs5', 'g.1hhc3m0jx', 'g.1hhc4j95p', 'm.0nfb4p8', 'g.12460pqpr', 'Ocho Rios', 'm.0nfb4rt', 'g.1hhc3x40g', 'g.1hhc4g963', 'm.0nf5r49', 'g.12tb6g_gq', 'm.0k5ntf6', 'Montego Bay', 'm.0nf75sk', 'm.0nf75_l', 'g.1245_v0g3', 'g.1hhc3r1_0', 'm.0nf4wl_', 'g.1hhc4lz6c', 'g.1245zrvvd', 'g.12cp_jvkq', 'g.124617_k_', 'g.1hhc4y0xy', 'm.0nf4whq', 'Country', 'm.0hny8h3', 'g.1245z_1xs', 'Kaliese Spencer', 'm.0nf75_t', 'g.1hhc3_4d2', 'm.0nf5q_g', 'g.1hhc3sjd_', 'g.11b60w91gb', 'g.1q5k9w5dr', 'g.1hhc3d9md', 'g.1hhc47p3n', 'Shericka Williams', 'g.1245zydws', 'g.12tb6flsg', 'Drumblair', 'g.124608jx8', 'g.1hhc3t8l3', 'g.1hhc3k0dh', 'g.1hhc4xfdg', 'g.1hhc4xmyt', 'g.1245zj344', 'g.12460s_80', 'm.0nf75xj', 'g.12460pbkm', 'g.1q5jj7s9l', 'g.12tb6gh4d', 'g.1hhc51psd', 'g.1hhc3w6gy', 'g.1hhc3xgjw', 'm.0k5nt58', 'g.1hhc4stjb', 'g.1hhc42f4w', 'g.1hhc3gxt5', 'm.0nf98kv', 'g.1245_wd9m', 'g.1hhc4stxy', 'g.1245zlvfv', 'm.0hny8q2', 'Steve Mullings', 'Simone Facey', 'g.1245_f_hv', 'g.1hhc4931t', 'm.07fwt_4', 'g.1245zx_sp', 'g.1245zdkh3', 'm.0hny8rj', 'm.0nf5r3w', 'g.1hhc4f812', 'g.1hhc4pgkb', 'g.1hhc3yzg0', 'g.1246170sh', 'g.11b60tf0bg', 'g.1hhc4t_n0', 'g.1hhc3yvdl', 'g.1hhc4bph3', 'g.1246098dl', 'g.124604q6p', 'm.0nf98g7', 'm.0nf75vz', 'g.1hhc3gls5', 'g.1hhc3sz8s', 'g.12460s647', 'g.1hhc4t0v6', 'g.1hhc3twwc', 'g.1hhc3pz1s', 'g.1245z5zsj', 'g.1hhc4z9_k', 'm.0nf5qzt', 'g.1hhc390mb', 'g.1hhc44k4b', 'm.0hny8sn', 'g.1q5jfl0lt', 'g.1hhc4cf_z', 'g.1hhc3twz9', 'g.11b71x3g2l', 'm.0nf4wkg', 'g.1245_4lvn', 'g.12tb6gtgp', 'g.11b60rrlc2']

parser = argparse.ArgumentParser()
parser.add_argument("--N_CPUS", type=int, default=mp.cpu_count())
parser.add_argument("--sample", type=int, default=-1)
parser.add_argument("--data_path", type=str, default="rmanluo")
parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
parser.add_argument("--save_cache", type=str, default="/data/shared/yuansui/rog/.cache/huggingface/datasets")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_path", type=str, default="results")
parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
parser.add_argument("--top_n", type=int, default=30)
parser.add_argument("--top_k", type=int, default=3)
parser.add_argument("--max_length", type=int, default=3)
parser.add_argument("--strategy", type=str, default="discrete_rating")
parser.add_argument("--squeeze", type=bool, default=True)
parser.add_argument("--verifier", type=str, default="enough")
parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
parser.add_argument("--add_hop_information", type=bool, default=True)
args = parser.parse_args()

llm_backbone = LLM_Backbone(args)
embeddings = llm_backbone.get_embeddings(x)

print(embeddings)