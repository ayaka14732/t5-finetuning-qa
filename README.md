# T5 Fine-Tuning on QA Dataset

The code in `lib/` is a direct fork of [ayaka14732/llama-jax](https://github.com/ayaka14732/llama-jax).

```sh
python3.11 -m venv venv
. venv/bin/activate
pip install -U pip
pip install -U wheel
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install git+https://github.com/huggingface/transformers.git
pip install -r requirements.txt
```

Model weights:

```sh
gdown 1f2sXC9wtpGm6vzIC0STDxWX8D7J1OTcM  # OmniDialog_T5basePreTrainingModel.tar.gz
tar -zxf OmniDialog_T5basePreTrainingModel.tar.gz  # base5/
rm -rf OmniDialog_T5basePreTrainingModel.tar.gz
```

Train:

```sh
export HF_HOME=/dev/shm/huggingface
python train.py
```

Inference:

```sh
gdown 1QIaajNStZEejqhx7M30p9MOp8qRYtSK4  # worldly-lion-48.npy
python inference.py
```

Serve in production:

Copy `env.json.example` to `env.json` and fill in your `deepl_apikey` and `openai_apikey`. Then run:

```sh
uvicorn --port 19230 serve:app
```
