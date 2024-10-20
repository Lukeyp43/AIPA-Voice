import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import aiohttp
import socketio
# Remove the eventlet import as we're not using it anymore
# import eventlet

# Add these new imports
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

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
from datetime import datetime
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.5, model_name="llama-3.1-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        # Add instruction to not assume responses are cut off
        system_prompt += "\nIMPORTANT: Do not assume user responses are cut off or incomplete. Respond to what is given without asking for completion."

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

async def get_transcript(callback, is_speaking, stop_flag):
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

        while not (transcription_complete.is_set() or stop_flag.is_set()):
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Error in get_transcript: {str(e)}")
        return None, None

    if stop_flag.is_set():
        print("Transcription stopped due to conversation end")
        return None, None

    if not transcription_end_time or not speech_end_time:
        print("Transcription did not complete successfully")
        return None, None

    return speech_end_time, transcription_end_time

class ConversationManager:
    def __init__(self, sio):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()
        self.tts = TextToSpeech()
        self.state = "IDLE"
        self.sio = sio
        self.stop_conversation_flag = asyncio.Event()
        self.end_phrases = ["have a nice day", "goodbye", "farewell"]
        self.summarizer = ConversationSummarizer()  # Initialize the summarizer

    def stop_all_processes(self):
        self.stop_conversation_flag.set()
        self.tts.stop_speaking()
        self.state = "IDLE"

    async def main(self, sid):
        try:
            await self.tts.init_session()
            self.stop_conversation_flag.clear()
            self.summarizer.clear_transcript()  # Clear any previous transcript
            
            # Start the conversation with an AI greeting
            initial_greeting = "Hello, welcome to our clinic my name is Miss PA. What is your name?"
            await self.sio.emit('ai_response', {'text': initial_greeting}, room=sid)
            await self.tts.speak(initial_greeting)
            self.summarizer.add_to_transcript("AI Assistant", initial_greeting)
            
            def handle_full_sentence(full_sentence):
                self.transcription_response = full_sentence

            while not self.stop_conversation_flag.is_set():
                print("Waiting for speech...")
                self.state = "LISTENING"
                await self.sio.emit('listening_status', {'isListening': True}, room=sid)
                speech_end_time, transcription_end_time = await get_transcript(
                    handle_full_sentence, 
                    lambda: False,
                    self.stop_conversation_flag
                )
                
                if self.stop_conversation_flag.is_set():
                    break

                if speech_end_time is None or transcription_end_time is None:
                    print("Error in speech recognition. Retrying...")
                    continue

                if not self.transcription_response:
                    print("No transcription received. Retrying...")
                    continue

                print(f"Transcription received: {self.transcription_response}")
                await self.sio.emit('transcription', {'text': self.transcription_response}, room=sid)

                # Add patient's response to summarizer
                self.summarizer.add_to_transcript("Patient", self.transcription_response)

                self.state = "PROCESSING"
                llm_response = self.llm.process(self.transcription_response)
                
                if not llm_response:
                    print("No response generated. Retrying...")
                    continue

                self.state = "TALKING"
                await self.sio.emit('ai_response', {'text': llm_response}, room=sid)
                speak_task = asyncio.create_task(self.tts.speak(llm_response))
                
                # Add AI's response to summarizer
                self.summarizer.add_to_transcript("AI Assistant", llm_response)
                
                await speak_task
                self.state = "LISTENING"
                self.transcription_response = ""

                # Check if the AI's response contains an end phrase
                if any(phrase in llm_response.lower() for phrase in self.end_phrases):
                    print("AI assistant ended the conversation.")
                    break

            print("Conversation ended.")
            
            # Generate and print summary at the end of conversation
            print("\nGenerating conversation summary...")
            summary, pdf_path, email_sent = self.summarizer.summarize_conversation(recipient_email="lukepettit10@gmail.com") 
            print("\nConversation Summary:")
            print(summary)
            print(f"\nPDF summary saved at: {pdf_path}")
            if email_sent:
                print("PDF sent via email successfully")
            else:
                print("Failed to send PDF via email")
            
            # Send summary to client
            await self.sio.emit('conversation_summary', {'summary': summary, 'pdf_path': pdf_path, 'email_sent': email_sent}, room=sid)

        finally:
            await self.tts.close_session()
            await self.sio.emit('listening_status', {'isListening': False}, room=sid)
            await self.sio.emit('conversation_ended', room=sid)

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

class ConversationSummarizer:
    def __init__(self):
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.conversation_transcript = []

    def add_to_transcript(self, role, content):
        """Add a new conversation entry to the transcript"""
        self.conversation_transcript.append((role, content))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def summarize_conversation(self, recipient_email=None):
        """
        Summarize the conversation with retry logic for API calls.
        Returns the generated summary as a string and the path to the PDF file.
        """
        # Format the conversation text with clear separation between speakers
        conversation_text = "\n".join([
            f"{role}:\n{content}\n" 
            for role, content in self.conversation_transcript
        ])

        prompt = f"""Analyze and summarize this medical conversation. Make sure to not put any * in the summary just plain text. Structure the summary as follows:

1. Patient Information:
   - Name
   - Primary concern

2. Symptom Details:
   - Description of symptoms
   - Onset and duration
   - Factors affecting symptoms (worsening or alleviating)

3. Medical History:
   - Relevant past conditions
   - Current medications

4. Key Points from AI Assistant's Questions:
   - Important follow-up questions asked
   - Significant information obtained

5. Next Steps:
   - Any immediate recommendations provided
   - Mention that the patient will see a doctor shortly

Format the summary with clear numbering and bullet points. Be concise but thorough.

Conversation:
{conversation_text}

Summary:"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.7,  # Add some variability while keeping it focused
                max_tokens=500    # Ensure we get a reasonably detailed summary
            )

            summary = chat_completion.choices[0].message.content.strip()
            pdf_path = self.generate_pdf(summary)
            
            if recipient_email:
                email_sent = self.send_pdf_email(pdf_path, recipient_email)
            else:
                email_sent = False

            return summary, pdf_path, email_sent

        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            error_summary = "Error: Unable to generate conversation summary"
            pdf_path = self.generate_pdf(error_summary)
            return error_summary, pdf_path, False

    def clear_transcript(self):
        """Clear the conversation transcript"""
        self.conversation_transcript = []

    def generate_pdf(self, summary):
        """Generate a PDF with the conversation summary"""
        # Create a directory for summaries if it doesn't exist
        summary_dir = "conversation_summaries"
        os.makedirs(summary_dir, exist_ok=True)

        # Generate a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(summary_dir, f"conversation_summary_{timestamp}.pdf")

        # Create PDF
        pdf = FPDF()
        pdf.add_page()

        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Conversation Summary", 0, 1, 'C')
        pdf.ln(10)

        # Add summary content
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, summary)

        # Save the PDF
        pdf.output(filename)
        print(f"PDF summary generated: {filename}")
        return filename

    def send_pdf_email(self, pdf_path, recipient_email):
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT"))

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Conversation Summary PDF"

        body = "Please find attached the conversation summary PDF."
        msg.attach(MIMEText(body, 'plain'))

        with open(pdf_path, "rb") as file:
            part = MIMEApplication(file.read(), Name=os.path.basename(pdf_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_path)}"'
        msg.attach(part)

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"PDF sent successfully to {recipient_email}")
            return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False

# Create a Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sio_app = socketio.ASGIApp(sio, app)

# Initialize the ConversationManager
conversation_manager = None

@sio.event
async def connect(sid, environ):
    print('Client connected:', sid)

@sio.event
async def disconnect(sid):
    print('Client disconnected:', sid)

@sio.event
async def start_conversation(sid):
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = ConversationManager(sio)
    await sio.start_background_task(conversation_manager.main, sid)

@sio.event
async def stop_conversation(sid):
    global conversation_manager
    if conversation_manager:
        conversation_manager.stop_all_processes()
        await sio.emit('listening_status', {'isListening': False}, room=sid)
        await sio.emit('conversation_ended', room=sid)
        conversation_manager = None  # Reset the conversation manager

if __name__ == "__main__":
    import uvicorn
    
    # Remove the dynamic port assignment and set it to 5001
    port = 5001
    uvicorn.run(sio_app, host="0.0.0.0", port=port)

