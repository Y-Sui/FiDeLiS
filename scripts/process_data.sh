
# # pre-process the dataset
# for dataset in RoG-cwq RoG-webqsp; do
#     for split in train; do
#         python src/align_kg/build_align_mcq_qa_dataset.py -d ${dataset} --split ${split} --sample 10
#         python src/joint_training/preprocess_align_mcq.py -d ${dataset} --sample 10
#     done
# done

python src/joint_training/preprocess_align_mcq.py -d RoG-webqsp --sample 10

python src/joint_training/preprocess_align_mcq.py -d RoG-cwq --sample 10