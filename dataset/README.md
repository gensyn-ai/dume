# Datasets setup

## Causal language modeling datasets

### OpenWebText Corpus dataset for pre-training

1. Clone this repository outside the root directory of the DUME repository: [HDEE: Heterogeneous Domain Expert Ensemble](https://github.com/gensyn-ai/hdee):

    ```
    git clone https://github.com/gensyn-ai/hdee
    ```

2. Install the required packages using ```pdm```, as described in the README of the HDEE repository.
3. Modify the ```hdee/scripts/load_openwebtext.sh``` script with your HuggingFace access token and ```meta-llama/Meta-Llama-3-8B``` as the tokenizer name.
4. Create the dataset directories:
    ```
    mkdir -p dataset/openwebtext
    ```
5. Open ```hdee/src/gensyn_dataprep/dataprep/pretokenize_data.py``` and substitute line 136 with the following line:
    ```
    parser.add_argument("--domain_name", action="store", type=str, required=False, default=None)
    ```
6. Execute the ```load_openwebtext.sh``` script from the ```HDEE``` repository root.
    ```
    chmod +x scripts/load_openwebtext.sh
    ./scripts/load_openwebtext.sh
    ```
7. Copy ```scripts/openwebtext_partitions.sh``` in ```[path/to/]hdee/dataset/openwebtext/raw``` and execute the script:
    ```
    cp scripts/openwebtext_partitions.sh [path/to/]hdee/dataset/openwebtext/raw
    cd [path/to/]hdee/dataset/openwebtext/raw
    chmod +x openwebtext_partitions.sh
    ./openwebtext_partitions.sh
    ```
8. Move the ```hdee/dataset/openwebtext``` directory in ```DUME/dataset```:
    ```
    cd [path/to/]DUME/
    mkdir -p dataset
    mv [path/to/]hdee/dataset/openwebtext dataset/  # or create a soft link with ln -s
    ```

### Domain experts datasets

1. Follow the steps for the OpenWebText Corpus dataset.
2. Modify the ```hdee/scripts/load_single_M2D2_domain_dataset.sh``` script with your HuggingFace access token and ```meta-llama/Meta-Llama-3-8B``` as the tokenizer name.
3. Execute the ```load_trained_M2D2_domains.sh``` script from the ```HDEE``` repository root.
    ```
    chmod +x scripts/load_trained_M2D2_domains.sh
    ./scripts/load_trained_M2D2_domains.sh
    ```
4. Move the datasets in ```\[cs_l1, History_and_events, math_l1, Philosophy_and_thinking, physics_l1\]``` in ```DUME/dataset```:
    ```
    mv [path/to/]hdee/dataset/[dataset] dataset/  # or create a soft link with ln -s
    ```

## Reasoning datasets

There is no need to download and configure the datasets manually, as this is handled automatically by the code.
