# Why you still see `max_new_tokens (=256)` in Colab

## 1. `generate_kwargs` on `GRPOTrainer` does **nothing**

TRL’s `GRPOTrainer` **does not** take `generate_kwargs=...`. If you add it, Python may error **or** it is dropped (Unsloth fork). The model still uses **`GRPOConfig.max_completion_length`**, which defaults to **256**.

**Fix:** **Delete** this line from your `GRPOTrainer(...)` call:

```python
generate_kwargs={"max_new_tokens": 512},  # DELETE - not a real TRL argument
```

## 2. Set length only on `GRPOConfig` (and re-run the cell)

```python
grpo_config = GRPOConfig(
    # ...
    max_completion_length=512,
    generation_kwargs={"max_new_tokens": 512},
)
# belt-and-suspenders after pull:
grpo_config.max_completion_length = 512
grpo_config.generation_kwargs = grpo_config.generation_kwargs or {}
grpo_config.generation_kwargs["max_new_tokens"] = 512
```

Then **Run all above** (or at least: config cell → trainer cell) so `grpo_config` in memory is really 512.

## 3. Verify after `GRPOTrainer` is created

```python
print("trainer.max_completion_length =", trainer.max_completion_length)
```

- If this prints **512** but logs still say **256** once, that can be a **stale warning** or a second code path.  
- If it prints **256**, your config was not applied (wrong/ old cell executed).

## 4. Unsloth + T4: extra cap (rare)

Some Unsloth + vLLM paths **lower effective max length** for memory (e.g. issues with `max_seq_len=256` in the engine). If step 3 shows **512** and training is fine, you can ignore a single Transformers **warning** about `max_length` vs `max_new_tokens`.

If you still see hard **256** in errors (not just a warning), try increasing Unsloth `gpu_memory_utilization` in `from_pretrained` per [Unsloth issues](https://github.com/unslothai/unsloth/issues/2215) or set `fast_inference=False` to avoid a smaller engine cap.

**Bottom line:** Remove `generate_kwargs` on the **trainer**, set **`max_completion_length=512` on the config**, restart runtime, run all cells top to bottom, check **`print(trainer.max_completion_length)`** == **512**.
