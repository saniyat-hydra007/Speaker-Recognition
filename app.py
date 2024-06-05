import os
import wave
from flask import Flask, render_template, request, redirect, url_for
import torchaudio
from speechbrain.inference import SpeakerRecognition
from werkzeug.utils import secure_filename

# Disable symlinks in Huggingface Hub cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = Flask(__name__)
app.config['RECORDINGS_FOLDER'] = 'recordings/'
app.config['AUDIO_FOLDER'] = 'audio/'
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

# Initialize the speaker verification model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['RECORDINGS_FOLDER'], filename)
        file.save(filepath)
        print(filepath)
        
        # Check the recorded audio against the existing audios
        match_found, match_filename = check_audio(filepath)
        
        if match_found:
            return render_template('success.html', match_filename=match_filename)
        else:
            save_new_audio(filepath)
            return redirect(url_for('index'))
    
    return redirect(request.url)

def check_audio(recorded_audio_path):
    max_score = 0
    match_filename = None
    
    for audio_file in os.listdir(app.config['AUDIO_FOLDER']):
        audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_file)
        print(audio_path)
        score, _ = verification.verify_files(recorded_audio_path, audio_path)
        print(score)
        if score > max_score:
            max_score = score
            match_filename = audio_file
    
    return max_score > 0.5, match_filename

def save_new_audio(filepath):
    existing_files = os.listdir(app.config['AUDIO_FOLDER'])
    existing_files = [f for f in existing_files if allowed_file(f)]
    max_id = max([int(os.path.splitext(f)[0]) for f in existing_files], default=0)
    new_filename = f"{max_id + 1}.wav"
    new_filepath = os.path.join(app.config['AUDIO_FOLDER'], new_filename)
    os.rename(filepath, new_filepath)

if __name__ == '__main__':
    os.makedirs(app.config['RECORDINGS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    app.run(debug=True)
