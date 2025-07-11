import os
import io
import json
import struct
import pygame
import pyaudio
import pvporcupine
import speech_recognition as sr
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq
from gtts import gTTS

# --- GÜVENLİ YAPILANDIRMA ve AYARLAR ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MEMORY_FILE = "memory.json"

# --- MODEL ve ASİSTAN AYARLARI ---
LLM_MODEL = "llama3-70b-8192"
WAKE_WORDS = ['jarvis', 'computer']
STOP_WORDS = ['computer']
TTS_SPEED = 1.0

# --- ARAÇLAR ve HAFIZA FONKSİYONLARI ---
tavily = TavilyClient(api_key=TAVILY_API_KEY)

def get_memory():
    if not os.path.exists(MEMORY_FILE) or os.path.getsize(MEMORY_FILE) == 0:
        return {}
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("Uyarı: memory.json dosyası bozuk, hafıza sıfırlandı.")
        return {}

def set_memory(key, value):
    memory = get_memory()
    memory[key] = value
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    return f"Not aldım: {key} = {value}"

# ## DEĞİŞTİRİLDİ ## - Hata yönetimi daha iyi hale getirildi.
def internet_search(query: str):
    """Güncel bilgi için internette arama yapar."""
    print(f"-> İnternet aranıyor: '{query}'")
    try:
        response = tavily.search(query=query, search_depth="basic")
        # Arama sonucu yoksa bunu belirtelim
        if not response['results']:
            return "İnternet aramasında bu konuyla ilgili bir sonuç bulamadım."
        return json.dumps([{"url": r["url"], "content": r["content"][:250]} for r in response['results']])
    except Exception as e:
        print(f"!!! Tavily Arama Hatası: {e}")
        return f"İnternet araması sırasında bir sorunla karşılaştım. API anahtarınızı kontrol etmeniz gerekebilir. Hata: {e}"

tools = [
    {"type": "function", "function": {"name": "internet_search", "description": "Güncel olaylar, hava durumu, haberler hakkında internette arama yap.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "set_memory", "description": "Kullanıcı hakkında önemli bir bilgiyi (ismi, tercihleri vb.) uzun süreli hafızaya kaydet.", "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}}
]

class VoiceAssistant:
    def __init__(self):
        # ... (bu kısımda değişiklik yok) ...
        self.client = Groq(api_key=GROQ_API_KEY)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        pygame.init(); pygame.mixer.init()
        try:
            self.porcupine = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keywords=WAKE_WORDS)
            self.stop_word_handle = pvporcupine.create(access_key=PICOVOICE_ACCESS_KEY, keywords=STOP_WORDS)
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(rate=self.porcupine.sample_rate, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=self.porcupine.frame_length)
        except Exception as e:
            print(f"Porcupine başlatılamadı: {e}"); exit()
        self.conversation_history = []
        self.tool_functions = {"internet_search": internet_search, "set_memory": set_memory}

    def stream_and_play_tts(self, text):
        # ... (bu kısımda değişiklik yok) ...
        print(f"Asistan: {text}")
        interrupted = False
        try:
            tts = gTTS(text=text, lang='tr', slow=False)
            audio_stream_obj = io.BytesIO()
            tts.write_to_fp(audio_stream_obj)
            audio_stream_obj.seek(0)
            pygame.mixer.music.load(audio_stream_obj, 'mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pcm = self.audio_stream.read(self.stop_word_handle.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.stop_word_handle.frame_length, pcm)
                if self.stop_word_handle.process(pcm) >= 0:
                    print("\n-- Durdurma komutu algılandı! --")
                    pygame.mixer.music.stop()
                    interrupted = True
                    break
                pygame.time.Clock().tick(30)
        except Exception as e:
            print(f"gTTS hatası veya dinleme sırasında hata: {e}")
        return interrupted

    # ## DEĞİŞTİRİLDİ ## - Timeout süresi artırıldı.
    def listen_and_transcribe_whisper(self):
        with sr.Microphone(sample_rate=16000) as source:
            print("\nDinliyorum...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Timeout süresi 5'ten 8 saniyeye çıkarıldı.
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=30)
                print("Anlaşılıyor...")
                transcript_text = self.recognizer.recognize_google(audio, language="tr-TR")
                print(f"Siz: {transcript_text}")
                return transcript_text
            except sr.UnknownValueError: return ""
            except sr.RequestError as e: print(f"Google servisinden sonuç alınamadı; {e}"); return ""
            except Exception as e: print(f"Dinleme sırasında bir hata oluştu: {e}"); return ""

    # ... (kodun geri kalanında değişiklik yok) ...
    def process_llm_response(self, user_input):
        current_memory = f"Kullanıcı hakkında bildiklerin: {json.dumps(get_memory(), ensure_ascii=False)}."
        system_prompt = f"Sen Asistan'sın, bir Türkçe sesli asistansın. Sohbet havasında, kısa ve net cevap ver. Bir araç (tool) sana bir hata mesajı döndürürse, bu mesajı direkt olarak kullanıcıya ilet. {current_memory}"
        if not self.conversation_history:
            self.conversation_history.append({"role": "system", "content": system_prompt})
        self.conversation_history.append({"role": "user", "content": user_input})
        response = self.client.chat.completions.create(model=LLM_MODEL, messages=self.conversation_history, tools=tools, tool_choice="auto")
        message = response.choices[0].message
        self.conversation_history.append(message)
        while message.tool_calls:
            for tool_call in message.tool_calls:
                function_to_call = self.tool_functions[tool_call.function.name]
                args = json.loads(tool_call.function.arguments)
                tool_response = function_to_call(**args)
                self.conversation_history.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name, "content": tool_response})
            response_with_tool_results = self.client.chat.completions.create(model=LLM_MODEL, messages=self.conversation_history)
            message = response_with_tool_results.choices[0].message
            self.conversation_history.append(message)
        return message.content

    def handle_conversation(self):
        self.stream_and_play_tts("Sizi dinliyorum.")
        self.conversation_history = []
        while True:
            user_input = self.listen_and_transcribe_whisper()
            if not user_input: continue
            if any(ext in user_input.lower() for ext in ["görüşürüz", "kapat", "uykuya geç"]):
                self.stream_and_play_tts("Görüşmek üzere."); break
            assistant_response = self.process_llm_response(user_input)
            was_interrupted = self.stream_and_play_tts(assistant_response)
            if was_interrupted: continue

    def run(self):
        print(f"Uyandırma sözcüğü bekleniyor ({', '.join(WAKE_WORDS)})...")
        print(f"Durdurma kelimeleri aktif ({', '.join(STOP_WORDS)}).")
        try:
            while True:
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False))
                if self.porcupine.process(pcm) >= 0:
                    print("\nUyandırma kelimesi algılandı!")
                    self.handle_conversation()
                    print(f"\n--- Uyku Modu ---")
                    print(f"Uyandırma sözcüğü bekleniyor ({', '.join(WAKE_WORDS)})...")
        finally: self.shutdown()

    def shutdown(self):
        if hasattr(self, 'porcupine'): self.porcupine.delete()
        if hasattr(self, 'stop_word_handle'): self.stop_word_handle.delete()
        if hasattr(self, 'audio_stream'): self.audio_stream.close()
        if hasattr(self, 'pa'): self.pa.terminate()
        pygame.quit()
        print("Asistan kapatıldı.")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()