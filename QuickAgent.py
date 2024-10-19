import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import aiohttp  # Add this line

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# Remove the ChatOpenAI import

from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from groq import InternalServerError  # Add this line

import time
from tenacity import retry, stop_after_attempt, wait_exponential

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process(self, text):
        self.memory.chat_memory.add_user_message(text)

        llm_start_time = time.time()
        try:
            response = self.conversation.invoke({"text": text})
        except InternalServerError as e:
            print(f"Groq API error: {e}.")
            return None  # Return None or handle the error as needed
        llm_end_time = time.time()

        self.memory.chat_memory.add_ai_message(response)

        llm_latency = int((llm_end_time - llm_start_time) * 1000)
        print(f"LLM Generation Latency: {llm_latency}ms")
        print(f"LLM Response: {response}")
        return response

class TextToSpeech:
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-asteria-en"

    def __init__(self):
        self.stop_event = asyncio.Event()
        self.session = None

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def speak(self, text):
        print("Starting TTS process...")
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {"text": text}

        player_process = await asyncio.create_subprocess_exec(
            "ffplay", "-autoexit", "-nodisp", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        tts_start_time = time.time()
        first_byte_time = None
        first_chunk_played = False

        try:
            await self.init_session()
            async with self.session.post(DEEPGRAM_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    print(f"Error response from Deepgram: {await response.text()}")
                    return

                async for chunk in response.content.iter_any():
                    if self.stop_event.is_set():
                        print("TTS interrupted")
                        break
                    if chunk:
                        if first_byte_time is None:
                            first_byte_time = time.time()
                            ttfb = int((first_byte_time - tts_start_time)*1000)
                            print(f"TTS Time to First Byte (TTFB): {ttfb}ms")
                        if not first_chunk_played:
                            first_chunk_time = time.time()
                            first_chunk_latency = int((first_chunk_time - tts_start_time)*1000)
                            print(f"TTS First Chunk Played Latency: {first_chunk_latency}ms")
                            first_chunk_played = True
                        player_process.stdin.write(chunk)
                        await player_process.stdin.drain()

        except Exception as e:
            print(f"Error during TTS process: {str(e)}")
        finally:
            if player_process.stdin:
                player_process.stdin.close()
            await player_process.wait()
            print("Finished TTS process")
            self.stop_event.clear()

    def stop_speaking(self):
        self.stop_event.set()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback, is_speaking):
    transcription_complete = asyncio.Event()
    transcription_end_time = None
    speech_end_time = None
    full_transcription = ""

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"), config)

        dg_connection = deepgram.listen.asyncwebsocket.v("1")
        print("Listening...")

        async def on_message(self, result, **kwargs):
            nonlocal full_transcription, transcription_end_time, speech_end_time
            try:
                sentence = result.channel.alternatives[0].transcript
                
                if len(sentence) == 0:
                    return

                print(f"Received partial transcription: {sentence}")

                if is_speaking():
                    callback(sentence.strip())  # Trigger interruption
                    return

                if result.is_final:
                    if result.speech_final:
                        full_transcription = sentence  # Use the final transcription
                        print(f"Human: {full_transcription}")
                        transcription_end_time = time.time()
                        speech_end_time = time.time()  # Set speech_end_time here
                        callback(full_transcription.strip())
                        transcription_complete.set()
            except Exception as e:
                print(f"Error in on_message: {str(e)}")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            smart_format=True,
            interim_results=True,
        )

        await dg_connection.start(options)

        microphone = Microphone(dg_connection.send)
        microphone.start()

        try:
            print("Waiting for transcription...")
            await asyncio.wait_for(transcription_complete.wait(), timeout=30)
        except asyncio.TimeoutError:
            print("Transcription timed out after 30 seconds")
        finally:
            print("Finishing microphone...")
            microphone.finish()
            print("Closing Deepgram connection...")
            await dg_connection.finish()

    except Exception as e:
        print(f"Error in get_transcript: {str(e)}")
        return None, None

    if not transcription_end_time or not speech_end_time:
        print("Transcription did not complete successfully")
        return None, None

    return speech_end_time, transcription_end_time

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.tts = TextToSpeech()
        self.state = "IDLE"
        self.interrupt_detected = False

    def handle_interrupt(self):
        if self.state == "TALKING":
            self.tts.stop_speaking()
            self.state = "LISTENING"
            self.interrupt_detected = True

    async def main(self):
        try:
            await self.tts.init_session()
            
            def handle_full_sentence(full_sentence):
                self.transcription_response = full_sentence
                if self.state == "TALKING":
                    self.handle_interrupt()

            while True:
                print("Waiting for speech...")
                self.state = "LISTENING"
                speech_end_time, transcription_end_time = await get_transcript(handle_full_sentence, lambda: self.state == "TALKING")
                
                if speech_end_time is None or transcription_end_time is None:
                    print("Error in speech recognition. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue

                if not self.transcription_response:
                    print("No transcription received. Retrying...")
                    continue

                print(f"Transcription received: {self.transcription_response}")  # Debug print

                if "goodbye" in self.transcription_response.lower():
                    break
                
                self.state = "PROCESSING"
                llm_response = self.llm.process(self.transcription_response)
                
                if not llm_response:
                    print("No response generated. Retrying...")
                    continue

                self.state = "TALKING"
                speak_task = asyncio.create_task(self.tts.speak(llm_response))
                
                while not speak_task.done():
                    if self.interrupt_detected:
                        print("Interruption detected, stopping TTS...")
                        self.tts.stop_speaking()
                        break
                    await asyncio.sleep(0.1)
                
                await speak_task
                self.state = "IDLE"
                self.interrupt_detected = False
                self.transcription_response = ""

            print("Conversation ended.")
        finally:
            await self.tts.close_session()

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())