# Learning Cluster Causal Diagrams using MCMC

### Install dependencies

```sh
pip install -r requirements.txt
```

### Run code

```sh
./run
```

### Options

Edit `run` to modify experiment parameters

|Option|Description|Type|Default|
|-|-|-|-|
|N_DATA_SAMPLES|Number of random synthetic data samples generated for training|`int`|1000|
|N_MCMC_SAMPLES|Number of C-DAG samples generated in one EM iteration|`int`|1000|
|N_MCMC_WARMUP|Number of warmup samples during one EM iteration|`int`|250|
|MAX_EM_ITERS|Number of EM steps|`int`|5|
|MAX_MLE_ITERS|Number of optimizer iterations for parameter optimization during one EM step|`int`|500|
|MIN_CLUSTERS|Minimum number of allowed clusters in a sampled clustering|`int`|2|
|MAX_CLUSTERS|Maximum number of allowed clusters in a sampled clustering|`int`|3|
