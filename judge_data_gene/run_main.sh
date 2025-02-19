
export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=llama3_ultrachat
model=/path/llama3_ultrachat
guidance_model=/path/llama3_70b_instruct
py_path=./judge_data_gene/main.py

input_path=./data/dolma_init_process.jsonl
few_shot_constraints_path=./data/few_shot_constraints.jsonl
few_shot_constraints_combine_path=./data/few_shot_constraints_combine.jsonl
output_path=./data/llm_as_a_judge_${model_name}.jsonl

# 润色
python ${py_path} \
    -i ${input_path} \
    -o ${output_path} \
    -m ${guidance_model} \
    -few_shot_constraints ${few_shot_constraints_path} \
    -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}

# 8b模型产生自己的回复
python ${py_path} \
    -i ${output_path} \
    -o ${output_path} \
    -m ${model} \
    -few_shot_constraints ${few_shot_constraints_path} \
    -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}


# 循环5次
for i in {1..5}; do
    echo "第${i}次循环"
    python ${py_path} \
        -i ${output_path} \
        -o ${output_path} \
        -m ${guidance_model} \
        -few_shot_constraints ${few_shot_constraints_path} \
        -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}

    python ${py_path} \
        -i ${output_path} \
        -o ${output_path} \
        -m ${model} \
        -few_shot_constraints ${few_shot_constraints_path} \
        -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}
    
    python ${py_path} \
        -i ${output_path} \
        -o ${output_path} \
        -m ${guidance_model} \
        -few_shot_constraints ${few_shot_constraints_path} \
        -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}
done


# 最后的处理
python ${py_path} \
    -i ${output_path} \
    -o ${output_path} \
    -m ${guidance_model} \
    -few_shot_constraints ${few_shot_constraints_path} \
    -few_shot_constraints_combine_path ${few_shot_constraints_combine_path}

