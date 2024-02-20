
# pre-process the dataset
for dataset in RoG-cwq; do
    for split in train; do
        python src/align_kg/build_align_qa_dataset.py -d ${dataset} --split ${split}
    done
done