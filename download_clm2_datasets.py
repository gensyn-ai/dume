"""Activate the trustds environment and run this script to download the datasets used in the CLM2 experts evaluation."""

from datasets import load_dataset


def main():

    datasets = [
        ("open-web-math/open-web-math", None),
        ("allenai/peS2o", None),
        ("pile-of-law/pile-of-law", "all"),
        ("HuggingFaceH4/ultrachat_200k", None)
    ]

    for arg1, arg2 in datasets:
        print(f"Loading dataset {arg1}...")
        try:
            load_dataset(arg1, arg2) if arg2 else load_dataset(arg1)
        except NotImplementedError:
            continue


if __name__ == '__main__':
    main()
