# GLM-4-Voice Server for UltraEval-Audio

This is the server for the UltraEval-Audio project.


# Setup

```shell
git clone https://github.com/UltraEval/GLM-4-Voice.git
cd GLM-4-Voice
conda create -n env python=3.10 -y
conda activate env
pip install -r requirments.txt
```

# Run
```shell
python adv_api_model_server.py
```

Now, you can use `--model glm-4-voice` in the UltraEval-Audio.