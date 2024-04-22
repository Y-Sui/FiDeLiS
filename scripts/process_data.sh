SOURCE_PATH="/data/shared/yuansui/rog"
export HF_HOME=${SOURCE_PATH}/.cache/huggingface

# pre-process the dataset
for dataset in RoG-cwq RoG-webqsp; do
    for split in train; do
      for sample in 5; do
         python src/align_kg/build_align_mcq_qa_dataset.py -d ${dataset} --split ${split} --sample ${sample}
         python src/joint_training/preprocess_align_mcq.py -d ${dataset} --sample ${sample}
      done
    done
done