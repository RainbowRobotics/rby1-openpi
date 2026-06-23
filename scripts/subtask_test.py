from PIL import Image
import numpy as np

import jax.numpy as jnp
import flax.nnx as nnx

from transformers import AutoTokenizer

from openpi.training import config as train_config
from openpi.models.pi0 import Pi0


PROMPT = (
    "Task: pick up the puppy in the middle.\n"
    "What should the robot do next?\n"
    "Subtask:"
)

IMAGE_PATH = "/home/ubuntu/sylee/openpi/scripts/puppy.jpg"
CONFIG_NAME = "pi05_libero" # debug_pi05 , pi05_libero
MAX_PROMPT_LEN = 200


def vocab_logits(model, hidden):
    embedding_table = model.PaliGemma.llm.embedder["input_embedding"].value
    return jnp.einsum("bd,vd->bv", hidden, embedding_table)


def generate_text_recompute(model, obs, tokenizer, max_new_tokens=30):
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)

    generated = []
    prompt_start, prompt_end = get_prompt_span(prefix_tokens, obs)


    for step in range(max_new_tokens):
        if generated:
            gen_ids = jnp.array([generated], dtype=jnp.int32)
            gen_emb = model.PaliGemma.llm(gen_ids, method="embed")

            tokens = jnp.concatenate([prefix_tokens, gen_emb], axis=1)
            input_mask = jnp.concatenate(
                [prefix_mask, jnp.ones(gen_ids.shape, dtype=bool)],
                axis=1,
            )
            ar_mask = jnp.concatenate(
                [
                    prefix_ar_mask,
                    jnp.ones((len(generated),), dtype=bool),
                ],
                axis=0,
            )
        else:
            tokens = prefix_tokens
            input_mask = prefix_mask
            ar_mask = prefix_ar_mask

        positions = jnp.cumsum(input_mask, axis=1) - 1
        attn_mask = make_attn_mask(input_mask, ar_mask)

        (out, _), _ = model.PaliGemma.llm(
            [tokens, None],
            mask=attn_mask,
            positions=positions,
        )

        if generated:
            hidden = out[:, -1, :]
        else:
            last_idx = prompt_end - 1
            hidden = out[:, last_idx, :]


        logits = vocab_logits(model, hidden)



        ####


        topk = jnp.argsort(logits[0])[-10:][::-1]

        print(f"\nSTEP {step}")
        print("TOP10")

        for idx in topk:
            idx = int(idx)
            print(
                idx,
                repr(tokenizer.decode([idx])),
                float(logits[0, idx]),
            )

        next_id = int(topk[0])

        print("CHOSEN:", repr(tokenizer.decode([next_id])))

        generated.append(next_id)


        ####


        # next_id = int(jnp.argmax(logits[0]))
        # generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("\n===== generated text =====")
    print(text)
    return generated, text



def make_attn_mask(input_mask, mask_ar):

    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)

    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]

    return jnp.logical_and(attn_mask, valid_mask)


def make_obs_with_image(obs, image_path):

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image = jnp.array(np.asarray(image), dtype=jnp.float32) / 255.0

    return obs.replace(
        images={
            name: image[None, ...]
            for name in obs.images.keys()
        }
    )


def set_prompt(obs, tokenizer, prompt):

    encoded = tokenizer(
        prompt,
        max_length=MAX_PROMPT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    return obs.replace(
        tokenized_prompt=jnp.array(encoded["input_ids"]),
        tokenized_prompt_mask=jnp.array(
            encoded["attention_mask"]
        ).astype(bool),
    )


def get_prompt_span(prefix_tokens, obs):
    prompt_len = int(jnp.sum(obs.tokenized_prompt_mask[0]))

    # 실제 prompt는 마지막 prompt_len개에 있음
    prompt_end = prefix_tokens.shape[1]
    prompt_start = prompt_end - prompt_len

    return prompt_start, prompt_end

def run_prefix(model, obs):


    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)


    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    # print("positions:", positions.shape)
    # print("last position:", positions[0, -1])

    attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

    # print("attn_mask:", attn_mask.shape)

    (prefix_out, _), _ = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=attn_mask,
        positions=positions,
    )

    # print("prefix_out:", prefix_out.shape)
    return prefix_out


def print_topk_next_token(model, tokenizer, hidden, k=20):
    logits = vocab_logits(model, hidden)
    topk = jnp.argsort(logits[0])[-k:][::-1]

    print("\nTOP-K NEXT TOKENS")
    for idx in topk:
        idx = int(idx)
        print(idx, repr(tokenizer.decode([idx])), float(logits[0, idx]))


def decode_prompt_hidden(model, tokenizer, prefix_out, obs):



    prompt_len = int(jnp.sum(obs.tokenized_prompt_mask[0]))

    prompt_start = (prefix_out.shape[1] - MAX_PROMPT_LEN)
    prompt_end = (prompt_start + prompt_len)

    print("prompt_start:", prompt_start)
    print("prompt_end:",prompt_end)

    prompt_hidden = prefix_out[:,prompt_start:prompt_end,:]

    print("prompt_hidden:", prompt_hidden.shape)

    embedding_table = model.PaliGemma.llm.embedder[
        "input_embedding"
    ].value

    logits = jnp.einsum(
        "btd,vd->btv",
        prompt_hidden,
        embedding_table,
    )

    tokens = jnp.argmax(logits, axis=-1)

    # 디버깅 프린트 있다가 지움
    encoded_prompt = obs.tokenized_prompt[0]

    prompt_tokens = encoded_prompt[-prompt_len:]

    print("PROMPT TOKENS")
    print(prompt_tokens)

    print(
        tokenizer.decode(
            [int(x) for x in prompt_tokens]
        )
    )


    print("decoded token ids:")
    print(tokens[0, :50])

    text = tokenizer.decode([int(x) for x in tokens[0]])

    print("\n===== decoded prompt hidden =====")
    print(text)

    last_hidden = prompt_hidden[:, -1, :]
    print_topk_next_token(model, tokenizer, last_hidden, k=20)



    return tokens


def compare_image_effect(model, obs_a, obs_b):

    prefix_out_a = run_prefix(model, obs_a)
    prefix_out_b = run_prefix(model, obs_b)

    prefix_tokens_a, _, _ = model.embed_prefix(obs_a)
    prompt_start_a, prompt_end_a = get_prompt_span(prefix_tokens_a, obs_a)

    prefix_tokens_b, _, _ = model.embed_prefix(obs_b)
    prompt_start_b, prompt_end_b = get_prompt_span(prefix_tokens_b, obs_b)

    prompt_a = prefix_out_a[:, prompt_start_a:prompt_end_a, :]
    prompt_b = prefix_out_b[:, prompt_start_b:prompt_end_b, :]

    diff = prompt_a - prompt_b

    mean_abs_diff = jnp.mean(jnp.abs(diff))
    max_abs_diff = jnp.max(jnp.abs(diff))

    cosine = jnp.sum(prompt_a * prompt_b, axis=-1)
    cosine = cosine / (
        jnp.linalg.norm(prompt_a, axis=-1)
        * jnp.linalg.norm(prompt_b, axis=-1)
        + 1e-8
    )

    print("mean abs diff:", mean_abs_diff)
    print("max abs diff:", max_abs_diff)
    print("mean cosine:", jnp.mean(cosine))

    return prompt_a, prompt_b


def main():

    # libero config 불러옴 
    cfg = train_config.get_config(CONFIG_NAME)

    rngs = nnx.Rngs(42)

    #pi05 모델 생성
    model = Pi0(cfg.model, rngs)

    # 파라미터 불러옴
    print("Loading weights...")

    state = nnx.state(model)
    params = state.to_pure_dict()

    loaded_params = cfg.weight_loader.load(params)

    nnx.update(model, loaded_params)

    embedding_table = model.PaliGemma.llm.embedder[
        "input_embedding"
    ].value


    print("Weights loaded.")

    # 팔리젬마 토크나이저 불러옴
    tokenizer = AutoTokenizer.from_pretrained(
        "google/paligemma-3b-pt-224"
    )

    # 가짜 데이터 생성 + 원하는 프롬프트 설정
    base_obs = cfg.model.fake_obs()
    base_obs = set_prompt(base_obs, tokenizer, PROMPT)

    puppy_obs = make_obs_with_image(base_obs, IMAGE_PATH)

    print("=" * 80)
    print("Config:", CONFIG_NAME)
    print("Prompt:", PROMPT)

    print("\n" + "=" * 80)
    print("Experiment 1: Decode prompt hidden")


    # 프롬프트 VLM output 디코딩
    prefix_out = run_prefix(model, puppy_obs)
    decode_prompt_hidden(model, tokenizer, prefix_out, puppy_obs)





    print("\n" + "=" * 80)
    print("Experiment 2: Image effect on prompt hidden")


    # 이미지가 프롬프트 히든에 미치는 영향 비교
    compare_image_effect(model, base_obs, puppy_obs)


    print("\n" + "=" * 80)
    print("Experiment 3: Autoregressive subtask generation")

    generate_text_recompute(
        model,
        puppy_obs,
        tokenizer,
        max_new_tokens=30,
    )


    print("\n" + "=" * 80)




if __name__ == "__main__":
    main()