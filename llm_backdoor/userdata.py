from datasets import load_dataset


def _load_hakurei_open_instruct_v1(sample_n: int):
    dataset = load_dataset("hakurei/open-instruct-v1", split="train")

    def _convert(row):
        prompt = row["instruction"]
        if row["input"]:
            prompt = f"{prompt}\n\n{row['input']}"
        return {"message": {"role": "user", "content": prompt}}

    return (
        dataset.shuffle()
        .select(range(sample_n))
        .map(
            _convert,
            remove_columns=["output", "input", "instruction"],
            desc="Converting",
        )
    )


LOAD_METHODS = {
    "hakurei/open-instruct-v1": _load_hakurei_open_instruct_v1,
}
