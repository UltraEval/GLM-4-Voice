import sys
sys.path.insert(0, "third_party/Matcha-TTS")
"""
A model worker executes the model.
"""
import argparse
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
import torch
import uvicorn
from transformers.generation.streamers import BaseStreamer
from threading import Thread
from queue import Queue
import os
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from flow_inference import AudioDecoder
import tempfile
import base64
from speech_tokenizer.utils import extract_speech_token


class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt

        # variables used in the streaming process
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class ModelWorker:
    def __init__(self, model_path, flow_path, tokenizer_path, device='cuda'):
        self.device = device
        self.glm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                                   device=device).to(device).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        flow_config = os.path.join(flow_path, "config.yaml")
        flow_checkpoint = os.path.join(flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(flow_path, 'hift.pt')

        # Flow & Hift
        self.audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                     hift_ckpt_path=hift_checkpoint,
                                     device=device)

        # Speech tokenizer
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.glm_tokenizer, self.glm_model

        prompt = params["prompt"]
        modal = params.get('modal', 's2t')
        audio_content = params.get("audio", None)
        if audio_content:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                data_buf = audio_content.encode("utf-8")
                f.write(base64.b64decode(data_buf))
                audio_tokens = extract_speech_token(
                    self.whisper_model, self.feature_extractor, [f.name]
                )[0]
                if len(audio_tokens) == 0:
                    raise ValueError("Audio file is too short to generate text")
                audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
                audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
                if modal == 's2t':
                    system_prompt = "User will provide you with a speech instruction. Think about the instruction and speak the response aloud directly."
                    prompt = "<|system|>\n{}<|user|>{}\n<|assistant|>transcript\n".format(system_prompt, audio_tokens)
                elif modal == "s2s":
                    system_prompt = "User will provide you with a speech instruction. Think about the instruction and speak the response aloud directly"
                    prompt = "<|system|>\n{}<|user|>{}\n<|assistant|>\n".format(system_prompt,
                                                                                              audio_tokens)
        temperature = float(params.get("temperature", 0.2))
        top_p = float(params.get("top_p", 0.8))
        max_new_tokens = int(params.get("max_new_tokens", 2000))

        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        streamer = TokenStreamer(skip_prompt=True)
        thread = Thread(target=model.generate,
                        kwargs=dict(**inputs, max_new_tokens=int(max_new_tokens),
                                    temperature=float(temperature), top_p=float(top_p),
                                    streamer=streamer))
        thread.start()

        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_token_id = self.glm_tokenizer.convert_tokens_to_ids('<|user|>')
        tts_mels = []
        prev_mel = None
        audio_tokens = []
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
        this_uuid = str(uuid.uuid4())
        text_tokens = []

        chunk_size = 20
        for token_id in streamer:
            is_finalize = token_id == end_token_id
            if not is_finalize:
                if token_id >= audio_offset:
                    token_id -= audio_offset
                    audio_tokens.append(token_id)
                else:
                    text_tokens.append(token_id)

            if prev_mel is not None:
                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

            if is_finalize:
                tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)

                tts_speech, tts_mel = self.audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                              prompt_token=flow_prompt_speech_token.to(self.device),
                                                              prompt_feat=prompt_speech_feat.to(self.device),
                                                              finalize=is_finalize)
                prev_mel = tts_mel
                tts_mels.append(tts_mel)
                audio_a = tts_speech.squeeze().cpu().numpy().tolist()
                for i in range(0, len(audio_a), 100):
                    yield (json.dumps({"token_id": token_id, "sample_rate": 22050,
                                       "text": self.glm_tokenizer.decode(text_tokens) if text_tokens else "",
                                       "audio": audio_a[i: i+100], "error_code": 0})).encode() + b"\0"
                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                audio_tokens = []

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Caught Unknown Error", e)
            ret = {
                "text": "Server Error",
                "error_code": 1,
            }
            yield (json.dumps(ret)+ "\n").encode()


app = FastAPI()


@app.post("/generate_stream")
async def generate_stream(request: Request):
    params = await request.json()

    generator = worker.generate_stream_gate(params)
    return StreamingResponse(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--flow-path", type=str, default="THUDM/glm-4-voice-decoder")
    parser.add_argument("--tokenizer", type=str, default="THUDM/glm-4-voice-tokenizer")
    args = parser.parse_args()

    worker = ModelWorker(args.model_path, args.flow_path, args.tokenizer)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
