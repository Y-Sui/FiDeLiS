
# pre-process the dataset
for dataset in RoG-cwq RoG-webqsp; do
    for split in train; do
        python src/align_kg/build_align_mcq_qa_dataset.py -d ${dataset} --split ${split} --sample 100
    done
done