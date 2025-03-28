"""
Tests for audio processing functionality in LlamaCanvas.

This module contains tests for audio features such as 
loading, manipulation, analysis, and conversion.
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from llama_canvas.audio import (
    load_audio,
    save_audio,
    get_audio_info,
    trim_audio,
    convert_audio_format,
    adjust_volume,
    mix_audios,
    extract_segment,
    add_fade,
    apply_audio_filter,
    generate_spectrogram,
    analyze_audio,
    detect_silence,
    detect_beats,
    transcribe_audio,
    audio_to_text
)


class TestAudioIO:
    """Tests for audio input/output functionality."""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create a sample audio data array for testing."""
        # Create a simple sine wave (2 seconds, 44.1kHz, mono)
        sample_rate = 44100
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a 440 Hz sine wave
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio_data, sample_rate
    
    @pytest.fixture
    def sample_audio_path(self, sample_audio_data, tmp_path):
        """Create a sample audio file and return its path."""
        audio_data, sample_rate = sample_audio_data
        audio_path = os.path.join(tmp_path, "sample.wav")
        
        with patch('llama_canvas.audio.sf') as mock_sf:
            save_audio(audio_data, audio_path, sample_rate=sample_rate)
            # Confirm save was attempted
            assert mock_sf.write.called
        
        return audio_path
    
    def test_load_audio(self, sample_audio_path):
        """Test loading audio files."""
        with patch('llama_canvas.audio.sf') as mock_sf:
            # Mock the return value of soundfile.read
            mock_sf.read.return_value = (np.zeros(44100), 44100)
            
            # Test loading with default parameters
            audio_data, sample_rate = load_audio(sample_audio_path)
            
            # Should call soundfile.read
            assert mock_sf.read.called
            
            # Should return audio data and sample rate
            assert isinstance(audio_data, np.ndarray)
            assert isinstance(sample_rate, int)
            
            # Test loading with mono conversion
            mock_sf.read.reset_mock()
            audio_mono, sr = load_audio(sample_audio_path, mono=True)
            
            # Should call soundfile.read
            assert mock_sf.read.called
            
            # Test loading with sample rate conversion
            mock_sf.read.reset_mock()
            audio_resampled, sr = load_audio(sample_audio_path, target_sr=22050)
            
            # Should still call soundfile.read
            assert mock_sf.read.called
            
            # Test with error handling
            mock_sf.read.side_effect = Exception("Error reading file")
            with pytest.raises(Exception):
                load_audio("nonexistent.wav")
    
    def test_save_audio(self, sample_audio_data):
        """Test saving audio data to file."""
        audio_data, sample_rate = sample_audio_data
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
            with patch('llama_canvas.audio.sf') as mock_sf:
                # Test saving with default parameters
                save_audio(audio_data, temp_path, sample_rate=sample_rate)
                
                # Should call soundfile.write
                assert mock_sf.write.called
                assert mock_sf.write.call_args[0][0] == temp_path
                assert np.array_equal(mock_sf.write.call_args[0][1], audio_data)
                assert mock_sf.write.call_args[0][2] == sample_rate
                
                # Test saving with different format
                mock_sf.write.reset_mock()
                save_audio(audio_data, temp_path, sample_rate=sample_rate, subtype="PCM_24")
                
                # Should call soundfile.write with subtype
                assert mock_sf.write.called
                assert mock_sf.write.call_args[1]["subtype"] == "PCM_24"
        
        os.unlink(temp_path)
    
    def test_get_audio_info(self, sample_audio_path):
        """Test retrieving audio file information."""
        with patch('llama_canvas.audio.sf') as mock_sf:
            # Mock SoundFile info
            mock_info = MagicMock()
            mock_info.samplerate = 44100
            mock_info.channels = 1
            mock_info.frames = 88200  # 2 seconds at 44.1kHz
            mock_info.format = "WAV"
            mock_info.subtype = "PCM_16"
            mock_info.duration = 2.0
            
            mock_sf.info.return_value = mock_info
            
            # Get audio info
            info = get_audio_info(sample_audio_path)
            
            # Should call sf.info
            assert mock_sf.info.called
            
            # Should return a dictionary with audio info
            assert isinstance(info, dict)
            assert info["samplerate"] == 44100
            assert info["channels"] == 1
            assert info["frames"] == 88200
            assert info["format"] == "WAV"
            assert info["subtype"] == "PCM_16"
            assert info["duration"] == 2.0
            
            # Test with non-existent file
            mock_sf.info.side_effect = Exception("File not found")
            with pytest.raises(Exception):
                get_audio_info("nonexistent.wav")


class TestAudioManipulation:
    """Tests for audio manipulation functions."""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create a sample audio data array for testing."""
        # Create a simple sine wave (2 seconds, 44.1kHz, mono)
        sample_rate = 44100
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a 440 Hz sine wave
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio_data, sample_rate
    
    def test_trim_audio(self, sample_audio_data):
        """Test trimming audio data."""
        audio_data, sample_rate = sample_audio_data
        
        # Test trimming with start and end times
        trimmed = trim_audio(audio_data, sample_rate, start_time=0.5, end_time=1.5)
        
        # Should return a numpy array
        assert isinstance(trimmed, np.ndarray)
        
        # Should be shorter than original
        assert len(trimmed) < len(audio_data)
        
        # Should have correct length (1 second of audio)
        expected_length = int(sample_rate * 1.0)  # 1 second
        assert len(trimmed) == expected_length
        
        # Test trimming with only start time
        trimmed_start = trim_audio(audio_data, sample_rate, start_time=1.0)
        assert len(trimmed_start) == int(sample_rate * 1.0)  # Last 1 second
        
        # Test trimming with only end time
        trimmed_end = trim_audio(audio_data, sample_rate, end_time=1.0)
        assert len(trimmed_end) == int(sample_rate * 1.0)  # First 1 second
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            trim_audio(audio_data, sample_rate, start_time=1.5, end_time=1.0)
        
        with pytest.raises(ValueError):
            trim_audio(audio_data, sample_rate, start_time=3.0)  # Beyond duration
    
    def test_convert_audio_format(self, sample_audio_data, tmp_path):
        """Test converting audio format."""
        audio_data, sample_rate = sample_audio_data
        
        # Create a sample audio file
        input_path = os.path.join(tmp_path, "input.wav")
        
        with patch('llama_canvas.audio.sf') as mock_sf:
            # Mock save to create input file
            save_audio(audio_data, input_path, sample_rate=sample_rate)
            
            # Test converting to mp3
            output_path = os.path.join(tmp_path, "output.mp3")
            
            with patch('llama_canvas.audio.subprocess') as mock_subprocess:
                # Mock subprocess run for ffmpeg conversion
                mock_process = MagicMock()
                mock_process.returncode = 0
                mock_subprocess.run.return_value = mock_process
                
                # Convert format
                result_path = convert_audio_format(input_path, output_format="mp3", output_path=output_path)
                
                # Should call subprocess.run with ffmpeg
                assert mock_subprocess.run.called
                cmd_args = mock_subprocess.run.call_args[0][0]
                assert "ffmpeg" in cmd_args
                assert "-i" in cmd_args
                assert input_path in cmd_args
                assert output_path in cmd_args
                
                # Should return output path
                assert result_path == output_path
                
                # Test with error
                mock_process.returncode = 1
                with pytest.raises(RuntimeError):
                    convert_audio_format(input_path, output_format="mp3")
                
                # Test with auto output path
                mock_process.returncode = 0
                result_path = convert_audio_format(input_path, output_format="flac")
                assert result_path.endswith(".flac")
    
    def test_adjust_volume(self, sample_audio_data):
        """Test adjusting audio volume."""
        audio_data, sample_rate = sample_audio_data
        
        # Test increasing volume
        amplified = adjust_volume(audio_data, gain_db=6.0)
        
        # Should return a numpy array
        assert isinstance(amplified, np.ndarray)
        
        # Should have same shape as input
        assert amplified.shape == audio_data.shape
        
        # Should be louder (higher amplitude)
        assert np.max(np.abs(amplified)) > np.max(np.abs(audio_data))
        
        # Test decreasing volume
        attenuated = adjust_volume(audio_data, gain_db=-6.0)
        
        # Should be quieter (lower amplitude)
        assert np.max(np.abs(attenuated)) < np.max(np.abs(audio_data))
        
        # Test with zero gain (should be unchanged)
        unchanged = adjust_volume(audio_data, gain_db=0.0)
        assert np.allclose(unchanged, audio_data)
    
    def test_mix_audios(self):
        """Test mixing multiple audio signals."""
        # Create two audio signals
        sample_rate = 44100
        duration = 1  # second
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # First audio: 440 Hz sine wave
        audio1 = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Second audio: 880 Hz sine wave
        audio2 = 0.3 * np.sin(2 * np.pi * 880 * t)
        
        # Test mixing with default weights
        mixed = mix_audios([audio1, audio2])
        
        # Should return a numpy array
        assert isinstance(mixed, np.ndarray)
        
        # Should have same length as inputs
        assert len(mixed) == len(audio1)
        
        # Test mixing with custom weights
        mixed_weighted = mix_audios([audio1, audio2], weights=[0.7, 0.3])
        
        # Should still have same length
        assert len(mixed_weighted) == len(audio1)
        
        # Test with inputs of different lengths
        audio3 = 0.2 * np.sin(2 * np.pi * 220 * t[:22050])  # Half duration
        
        # Should raise error for different lengths without padding
        with pytest.raises(ValueError):
            mix_audios([audio1, audio3], pad=False)
        
        # Should work with padding
        mixed_padded = mix_audios([audio1, audio3], pad=True)
        assert len(mixed_padded) == len(audio1)
    
    def test_extract_segment(self, sample_audio_data):
        """Test extracting audio segments."""
        audio_data, sample_rate = sample_audio_data
        
        # Test extracting a segment
        start_time = 0.5  # seconds
        end_time = 1.0  # seconds
        segment = extract_segment(audio_data, sample_rate, start_time, end_time)
        
        # Should return a numpy array
        assert isinstance(segment, np.ndarray)
        
        # Should have correct length
        expected_length = int(sample_rate * (end_time - start_time))
        assert len(segment) == expected_length
        
        # Test with invalid times
        with pytest.raises(ValueError):
            extract_segment(audio_data, sample_rate, 1.5, 1.0)  # start > end
        
        with pytest.raises(ValueError):
            extract_segment(audio_data, sample_rate, -0.5, 1.0)  # start < 0
        
        with pytest.raises(ValueError):
            extract_segment(audio_data, sample_rate, 0.5, 3.0)  # end > duration
    
    def test_add_fade(self, sample_audio_data):
        """Test adding fade-in and fade-out effects."""
        audio_data, sample_rate = sample_audio_data
        
        # Test adding fade-in
        fade_in = add_fade(audio_data, sample_rate, fade_in_sec=0.5, fade_out_sec=0)
        
        # Should return a numpy array
        assert isinstance(fade_in, np.ndarray)
        
        # Should have same shape as input
        assert fade_in.shape == audio_data.shape
        
        # First sample should be close to zero
        assert abs(fade_in[0]) < 0.01
        
        # Test adding fade-out
        fade_out = add_fade(audio_data, sample_rate, fade_in_sec=0, fade_out_sec=0.5)
        
        # Last sample should be close to zero
        assert abs(fade_out[-1]) < 0.01
        
        # Test adding both fade-in and fade-out
        fade_both = add_fade(audio_data, sample_rate, fade_in_sec=0.3, fade_out_sec=0.3)
        
        # First and last samples should be close to zero
        assert abs(fade_both[0]) < 0.01
        assert abs(fade_both[-1]) < 0.01
        
        # Test with fade longer than audio
        with pytest.raises(ValueError):
            add_fade(audio_data, sample_rate, fade_in_sec=1.5, fade_out_sec=1.5)  # Total > duration
    
    def test_apply_audio_filter(self, sample_audio_data):
        """Test applying filters to audio."""
        audio_data, sample_rate = sample_audio_data
        
        with patch('llama_canvas.audio.scipy.signal') as mock_signal:
            # Mock filter design
            mock_signal.butter.return_value = ([0.5], [1.0])
            
            # Mock filter application
            mock_signal.filtfilt.return_value = audio_data * 0.8  # Attenuated signal
            
            # Test lowpass filter
            filtered = apply_audio_filter(audio_data, sample_rate, filter_type="lowpass", cutoff_freq=1000)
            
            # Should call filter design and application
            assert mock_signal.butter.called
            assert mock_signal.filtfilt.called
            
            # Should return filtered signal
            assert isinstance(filtered, np.ndarray)
            assert filtered.shape == audio_data.shape
            
            # Test highpass filter
            mock_signal.butter.reset_mock()
            mock_signal.filtfilt.reset_mock()
            
            filtered = apply_audio_filter(audio_data, sample_rate, filter_type="highpass", cutoff_freq=500)
            
            # Should call filter with highpass type
            assert mock_signal.butter.called
            filter_args = mock_signal.butter.call_args[0]
            assert filter_args[2] == "highpass"
            
            # Test bandpass filter
            mock_signal.butter.reset_mock()
            filtered = apply_audio_filter(
                audio_data, sample_rate, 
                filter_type="bandpass", 
                cutoff_freq=[500, 2000]
            )
            
            # Should call filter with bandpass type
            assert mock_signal.butter.called
            filter_args = mock_signal.butter.call_args[0]
            assert filter_args[2] == "bandpass"
            
            # Test invalid filter type
            with pytest.raises(ValueError):
                apply_audio_filter(audio_data, sample_rate, filter_type="invalid")
            
            # Test invalid cutoff frequency
            with pytest.raises(ValueError):
                apply_audio_filter(audio_data, sample_rate, filter_type="bandpass", cutoff_freq=1000)


class TestAudioAnalysis:
    """Tests for audio analysis functions."""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create a sample audio data array for testing."""
        # Create a simple sine wave (2 seconds, 44.1kHz, mono)
        sample_rate = 44100
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a 440 Hz sine wave
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio_data, sample_rate
    
    def test_generate_spectrogram(self, sample_audio_data):
        """Test generating spectrograms from audio."""
        audio_data, sample_rate = sample_audio_data
        
        with patch('llama_canvas.audio.plt') as mock_plt, \
             patch('llama_canvas.audio.librosa.display') as mock_display:
            
            # Test generating spectrogram with default parameters
            spec = generate_spectrogram(audio_data, sample_rate)
            
            # Should return a numpy array
            assert isinstance(spec, np.ndarray)
            
            # Should have 2D shape (frequency bins × time frames)
            assert len(spec.shape) == 2
            
            # Test saving spectrogram to file
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                mock_plt.savefig.return_value = None
                
                spec_path = generate_spectrogram(
                    audio_data, sample_rate, 
                    output_path=temp_file.name,
                    as_image=True
                )
                
                # Should call matplotlib to save figure
                assert mock_plt.savefig.called
                
                # Should return path to image
                assert spec_path == temp_file.name
            
            # Test with different parameters
            spec_custom = generate_spectrogram(
                audio_data, sample_rate,
                n_fft=1024,
                hop_length=256,
                fmin=50,
                fmax=8000
            )
            
            # Should still return a numpy array
            assert isinstance(spec_custom, np.ndarray)
    
    def test_analyze_audio(self, sample_audio_data):
        """Test comprehensive audio analysis."""
        audio_data, sample_rate = sample_audio_data
        
        # Test analyzing audio with default parameters
        analysis = analyze_audio(audio_data, sample_rate)
        
        # Should return a dictionary with analysis results
        assert isinstance(analysis, dict)
        assert "duration" in analysis
        assert "rms" in analysis
        assert "spectral_centroid" in analysis
        assert "spectral_bandwidth" in analysis
        assert "spectral_rolloff" in analysis
        assert "zero_crossing_rate" in analysis
        
        # Check duration
        assert np.isclose(analysis["duration"], 2.0)
        
        # Test with different parameters (e.g., frame size)
        analysis_custom = analyze_audio(audio_data, sample_rate, frame_size=2048, hop_length=1024)
        
        # Should still return a dictionary
        assert isinstance(analysis_custom, dict)
        
        # Duration should be the same
        assert np.isclose(analysis_custom["duration"], analysis["duration"])
    
    def test_detect_silence(self, sample_audio_data):
        """Test silence detection in audio."""
        audio_data, sample_rate = sample_audio_data
        
        # Create audio with silent segments
        # First half second and last half second are silent
        audio_with_silence = audio_data.copy()
        silent_samples = int(sample_rate * 0.5)
        audio_with_silence[:silent_samples] = 0
        audio_with_silence[-silent_samples:] = 0
        
        # Test detecting silence
        silence_intervals = detect_silence(audio_with_silence, sample_rate)
        
        # Should return a list of intervals
        assert isinstance(silence_intervals, list)
        
        # Should find two silent intervals
        assert len(silence_intervals) == 2
        
        # Each interval should be a tuple of (start_time, end_time)
        assert isinstance(silence_intervals[0], tuple)
        assert len(silence_intervals[0]) == 2
        
        # Test with different threshold
        silence_high_thresh = detect_silence(audio_with_silence, sample_rate, threshold=0.1)
        
        # With higher threshold, should find more silence
        assert len(silence_high_thresh) >= len(silence_intervals)
        
        # Test with minimum duration
        silence_min_dur = detect_silence(audio_with_silence, sample_rate, min_duration=1.0)
        
        # Should filter out short silences
        assert len(silence_min_dur) < len(silence_intervals)
    
    def test_detect_beats(self, sample_audio_data):
        """Test beat detection in audio."""
        audio_data, sample_rate = sample_audio_data
        
        with patch('llama_canvas.audio.librosa.beat') as mock_beat:
            # Mock beat detection result
            mock_beat.beat_track.return_value = (120.0, np.array([0.5, 1.0, 1.5]))
            
            # Test detecting beats
            tempo, beat_times = detect_beats(audio_data, sample_rate)
            
            # Should call librosa.beat.beat_track
            assert mock_beat.beat_track.called
            
            # Should return tempo and beat times
            assert isinstance(tempo, float)
            assert tempo == 120.0
            assert isinstance(beat_times, np.ndarray)
            assert len(beat_times) == 3
            
            # Test with custom parameters
            mock_beat.beat_track.reset_mock()
            tempo, beat_times = detect_beats(audio_data, sample_rate, start_bpm=90)
            
            # Should call beat_track with start_bpm
            assert mock_beat.beat_track.called
            assert mock_beat.beat_track.call_args[1]["start_bpm"] == 90


class TestAudioTranscription:
    """Tests for audio transcription functions."""
    
    @pytest.fixture
    def sample_audio_path(self, tmp_path):
        """Create a sample audio file path."""
        return os.path.join(tmp_path, "speech.wav")
    
    def test_transcribe_audio(self, sample_audio_path):
        """Test transcribing audio to text."""
        with patch('llama_canvas.audio.sr') as mock_sr:
            # Mock recognition
            mock_sr.Recognizer.return_value.recognize_google.return_value = "This is a test transcription"
            mock_sr.AudioFile.return_value.__enter__.return_value = "audio_data"
            mock_sr.Recognizer.return_value.record.return_value = "audio_record"
            
            # Test transcribing with default parameters
            transcription = transcribe_audio(sample_audio_path)
            
            # Should return transcription text
            assert isinstance(transcription, str)
            assert transcription == "This is a test transcription"
            
            # Test with different language
            mock_sr.Recognizer.return_value.recognize_google.reset_mock()
            mock_sr.Recognizer.return_value.recognize_google.return_value = "C'est un test"
            
            transcription_fr = transcribe_audio(sample_audio_path, language="fr-FR")
            
            # Should call recognize_google with language
            assert mock_sr.Recognizer.return_value.recognize_google.called
            assert mock_sr.Recognizer.return_value.recognize_google.call_args[1]["language"] == "fr-FR"
            
            # Test with recognition error
            mock_sr.Recognizer.return_value.recognize_google.side_effect = mock_sr.UnknownValueError()
            
            # Should return empty string or error message
            assert transcribe_audio(sample_audio_path) == ""
    
    def test_audio_to_text(self, sample_audio_path):
        """Test high-level audio to text conversion."""
        with patch('llama_canvas.audio.transcribe_audio') as mock_transcribe:
            # Mock transcription function
            mock_transcribe.return_value = "This is a test transcription"
            
            # Test converting audio to text
            text, confidence = audio_to_text(sample_audio_path)
            
            # Should call transcribe_audio
            assert mock_transcribe.called
            
            # Should return text and confidence
            assert isinstance(text, str)
            assert text == "This is a test transcription"
            assert isinstance(confidence, float)
            
            # Test with different engine
            mock_transcribe.reset_mock()
            text, confidence = audio_to_text(sample_audio_path, engine="whisper")
            
            # Should call transcribe_audio with engine
            assert mock_transcribe.called
            assert mock_transcribe.call_args[1]["engine"] == "whisper"


if __name__ == "__main__":
    pytest.main() 