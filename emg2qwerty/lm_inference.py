from __future__ import annotations

import abc

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field, InitVar
from typing import Any, ClassVar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import kenlm
import numpy as np

from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.data import LabelData
import openai

def gpt_autocorrect(sentence: str, model="gpt-4o-mini") -> str:
    client = openai.OpenAI(api_key = "API-KEY")
    prompt = (
        "Correct the spelling *if needed* and return only the corrected word.\n"
        "Do not include punctuation or explanation.\n"
        "Do not change the case of the word.\n\n"
        f"{sentence.strip()}"
    )
    print(f"[DEBUG] Sending prompt to GPT: '{sentence.strip()}'")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"[DEBUG] GPT response: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        return sentence


@dataclass
class CTCBeamDecoderWithGPTWord(CTCBeamDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        words = clean_text.split()
        corrected_words = []
        for word in words:
            corrected_words.append(gpt_autocorrect(word))
        label_data.text = ' '.join(corrected_words)
        return label_data

@dataclass
class CTCBeamDecoderWithGPTSentence(CTCBeamDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        corrected = gpt_autocorrect(clean_text)
        label_data.text = corrected
        return label_data

@dataclass
class CTCGreedyDecoderWithGPTWord(CTCGreedyDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        words = clean_text.split()
        corrected_words = []
        for word in words:
            corrected_words.append(gpt_autocorrect(word))
        label_data.text = ' '.join(corrected_words)
        return label_data
@dataclass
class CTCGreedyDecoderWithGPTSentence(CTCGreedyDecoder):
    @staticmethod
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol and result:
                result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(self, emissions: np.ndarray, timestamps: np.ndarray, finish: bool = False) -> LabelData:
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        if not raw_text:
            return label_data

        clean_text = self.apply_backspaces(raw_text)
        corrected = gpt_autocorrect(clean_text)
        label_data.text = corrected
        return label_data
      
class FlanT5Autocorrector:
    def __init__(self, model_name="google/flan-t5-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def correct(self, text: str) -> str:
        print(f"[DEBUG] Autocorrecting text: '{text}'")
        prompt = f"Correct this sentence for spelling errors only. Keep valid words unchanged:\n{text.strip()}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=32)  # allow more tokens for phrases

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return decoded if decoded else text


@dataclass
class CTCBeamDecoderWithFlan(CTCBeamDecoder):
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol:
                if result:
                    result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(
        self,
        emissions: np.ndarray,
        timestamps: np.ndarray,
        finish: bool = False,
    ) -> LabelData:
        print("[DEBUG] Using CTCBeamDecoderWithFlan.decode()")

        # Step 1: Run standard beam search
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        print(f"[DEBUG] Raw decoded text (pre-backspace): '{raw_text}'")

        if not raw_text:
            print("[DEBUG] Empty text after beam decoding. Returning early.")
            return label_data

        # Step 2: Handle backspaces
        clean_text = CTCBeamDecoderWithFlan.apply_backspaces(raw_text)
        print(f"[DEBUG] Text after backspace handling: '{clean_text}'")

        # Step 3: Sentence-level autocorrection using FlanT5
        try:
            corrected = gpt_autocorrect(clean_text)
            label_data.text = corrected
            print(f"[DEBUG] Text after GPT correction: '{corrected}'")
            label_data.text = corrected
        except Exception as e:
            print(f"⚠️ [ERROR] GPT correction failed: {e}")
            label_data.text = clean_text  # Fallback

        return label_data

    def decode_batch(
        self,
        emissions: np.ndarray,
        emission_lengths: np.ndarray,
    ) -> list[LabelData]:
        print("[DEBUG] Using CTCBeamDecoderWithFlan.decode_batch()")
        assert emissions.ndim == 3
        assert emission_lengths.ndim == 1
        N = emissions.shape[1]

        decodings = []
        for i in range(N):
            self.reset()
            decoded = self.decode(
                emissions=emissions[: emission_lengths[i], i],
                timestamps=np.arange(emission_lengths[i]),
            )
            
            decodings.append(decoded)

        return decodings


@dataclass
class CTCBeamDecoderWithFlanWordLevel(CTCBeamDecoder):
    def apply_backspaces(text: str, backspace_symbol: str = "⌫") -> str:
        result = []
        for char in text:
            if char == backspace_symbol:
                if result:
                    result.pop()
            else:
                result.append(char)
        return ''.join(result)

    def decode(
        self,
        emissions: np.ndarray,
        timestamps: np.ndarray,
        finish: bool = False,
    ) -> LabelData:
        print("🧠 [DEBUG] Running CTCBeamDecoderWithFlanWordLevel.decode()")

        # Step 1: Run standard beam search
        label_data = super().decode(emissions, timestamps, finish)
        raw_text = label_data.text.strip()
        print(f"[DEBUG] Raw decoded text (pre-backspace): '{raw_text}'")

        if not raw_text:
            print("[DEBUG] Empty decoded text. Returning early.")
            return label_data

        # Step 2: Handle backspaces
        clean_text = CTCBeamDecoderWithFlanWordLevel.apply_backspaces(raw_text)
        print(f"[DEBUG] Text after backspace handling: '{clean_text}'")

        # Step 3: Autocorrect word-by-word
        autocorrector = FlanT5Autocorrector()
        buffer = []
        corrected_words = []

        for ch in clean_text:
            buffer.append(ch)
            if ch == " ":
                word = ''.join(buffer).strip()
                if word:
                    try:
                        corrected = autocorrector.correct(word)
                        print(f"[DEBUG] Word '{word}' → '{corrected}'")
                        corrected_words.append(corrected)
                    except Exception as e:
                        print(f"⚠️ [ERROR] Flan error on word '{word}': {e}")
                        corrected_words.append(word)
                buffer = []

        # Final word flush
        if buffer:
            word = ''.join(buffer).strip()
            if word:
                try:
                    corrected = autocorrector.correct(word)
                    print(f"[DEBUG] Final word '{word}' → '{corrected}'")
                    corrected_words.append(corrected)
                except Exception as e:
                    print(f"⚠️ [ERROR] Flan error on final word '{word}': {e}")
                    corrected_words.append(word)

        label_data.text = ' '.join(corrected_words)
        print(f"✅ [DEBUG] Final corrected text: '{label_data.text}'")

        return label_data
