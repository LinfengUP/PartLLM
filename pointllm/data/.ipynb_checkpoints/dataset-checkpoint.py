import glob
import os
import random

import torch
from pointllm import conversation as conversation_lib

DEFAULT_POINT_END_TOKEN = "<point_end>"
DEFAULT_POINT_PATCH_TOKEN = "<point_patch>"
DEFAULT_POINT_START_TOKEN = "<point_start>"
DEATULT_POINT_TOKEN = "<point>"
POINT_TOKEN_INDEX = -200
IGNORE_INDEX = -100


def collate_fn(
        batch,tokenizer=None,conv_type="llava_v1",use_mm_start_end=True,local_rank=-1
):
    point_path_list = []
    point_cloud_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    question_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for(
        point_path,
        point_cloud,
        conversations,
        masks,
        label,
        questions,
        sample_classes,
        inferences,
    ) in batch:
        point_path_list.append(point_path)
        point_cloud_list.append(point_cloud)
        conversation_list.append(conversations)
        masks_list.append(masks)
        label_list.append(label)
        question_list.append(questions)
        sampled_classes_list.append(sample_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inferences)

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEATULT_POINT_TOKEN
            replace_token = (
                DEFAULT_POINT_START_TOKEN + replace_token + DEFAULT_POINT_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEATULT_POINT_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_point_token(prompt,tokenizer,return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            
            assert len(parts)==2, (len(parts),rou)
            parts[0] += sep
    
            if DEATULT_POINT_TOKEN in conversation:
                round_len = len(tokenizer_point_token(rou,tokenizer))
                instruction_len = len(tokenizer_point_token(parts[0],tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len:cur_len+instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z==IGNORE_INDEX,tokenizer.unk_token_id,z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )
        
        if cur_len<tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length-255
        
        if input_ids.shape[1]>truncate_len:
            input_ids = input_ids[:,:truncate_len]
            targets = targets[:,:truncate_len]
            attention_mask = attention_mask[:,:truncate_len]
        
    return{
        "point_paths": point_path_list,
        "point_clouds": point_cloud_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "masks_list": masks_list,
        "label_list": label_list,
        "offset": torch.LongTensor(offset_list),
        "question_list": question_list,
        "sampled_classes_list": sampled_classes_list,
        "inferences": inferences[0],
        "conversation_list": conversation_list,
    }







def tokenizer_point_token(
    prompt, tokenizer, point_token_index=POINT_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<point>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [point_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids
