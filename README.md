# T5 Fine-Tuning on QA Dataset

The code in `lib/` is a direct fork of [ayaka14732/llama-jax](https://github.com/ayaka14732/llama-jax).

```sh
python3.11 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

Model weights:

```sh
gdown 1f2sXC9wtpGm6vzIC0STDxWX8D7J1OTcM  # OmniDialog_T5basePreTrainingModel.tar.gz
tar -zxf OmniDialog_T5basePreTrainingModel.tar.gz  # base5/
rm -rf OmniDialog_T5basePreTrainingModel.tar.gz
```
