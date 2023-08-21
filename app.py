import gradio as gr
import sounddevice as sd
import wavio
import whisper
import numpy as np
from gtts import gTTS
from slyme import SlymeDriver
import time
import re


base = whisper.load_model("base")
small = whisper.load_model("small")
medium = whisper.load_model("medium")

slyme = SlymeDriver(pfname='Default')
time.sleep(5)
slyme.select_latest_chat()
time.sleep(5)

def speech_to_text(audio_file, model_option):
    model = base
    if model_option == "small":
        model = small
    elif model_option == "medium":
        model = medium
    result = model.transcribe(audio_file)
    print(result)
    message = result["text"]
    lang = result["language"]
    return message, lang


def generate_answer(message):
    prompt = message
    #output = slyme.completion(prompt)
    output = slyme.completion(prompt, interval=0.7)
    print(output)

    # slyme.end_session()
    return output


def text_to_speech(audio_file, text_data, lang):
    myobj = gTTS(text=text_data, lang=lang, slow=False)
    repertoire = re.search(r"^(.+)/[^/]+$", audio_file).group(1)
    file = repertoire+"/reponse.wav"
    myobj.save(file)
    return file


def voice_bot(model_option, audio_data):
    question, lang = speech_to_text(audio_data, model_option)
    answer = generate_answer(question)
    audio_data = text_to_speech(audio_data, answer, lang)
    return question, answer, audio_data

# iface = gr.Interface(
#     fn=speech_to_text,
#     inputs=[gr.Radio(["base", "small", "medium"]),
#             gr.Audio(source="microphone", type="filepath", label="Capture Audio")],
#     outputs=["text", "text"],
#     live=True,
# )


#iface = gr.Interface(
#      fn=generate_answer,
#      inputs="text",
#      outputs="text",
#)


# iface = gr.Interface(
#     fn=text_to_speech,
#     inputs=["text", "text", "text"],
#     outputs=gr.Audio(),
# )


#iface = gr.Interface(
#    fn=voice_bot,
#    inputs=gr.Audio(source="microphone", type="filepath", label="Capture Audio"),
#    outputs=gr.Audio(),
#)


#iface = gr.Interface(
#    title="Voice-Bot",
#    description="To ask your question, click on 'record from microphone', then 'stop recording' before 'submit'",
#    fn=voice_bot,
#    inputs=[gr.Radio(["base", "small", "medium"], info="Choice of Speech-To-Text model"),
#            gr.Audio(source="microphone", type="filepath", label="1. Capture Audio")],
#    outputs=[gr.Textbox(label="2. Transcription of the question"), 
#             gr.Textbox(label="3. Transcription of the answer"), 
#             gr.Audio(label="4. Audio answer", type="filepath")]
#)
#


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Voice-Bot
    To ask your question, click on 'record from microphone', then 'stop recording' before 'submit'
    """
    )
    input1 = gr.Radio(["base", "small", "medium"], label="Choice of Speech-To-Text model")
    input2 = gr.Audio(source="microphone", type="filepath", label="1. Capture Audio")

    btn = gr.Button(value="Submit")
    
    output1 = gr.Textbox(label="2. Transcription of the question")

    gr.Markdown(
        """
    ## Outputs
    """
    )
    output2 = gr.Textbox(label="3. Transcription of the answer")
    output3 = gr.Audio(label="4. Audio answer", type="filepath")

    btn.click(voice_bot, inputs=[input1, input2], outputs=[output1, output2, output3])


if __name__ == "__main__":
    demo.launch(share=True)


#if __name__ == "__main__":
#    iface.launch(share=True)
