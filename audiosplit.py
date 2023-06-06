#!/usr/bin/env python3

import collections
import contextlib
from genericpath import exists
import sys
import wave
import os
import webrtcvad
import soundfile
import argparse

MIN_SEG_LEN = 5

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames, minimal_segment_duration, maximal_segment_duration):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)


    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    st = 0
    en = 0
    last_frame_en = 0
    voiced_frames = []
    for fn, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        
        #sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                #sys.stdout.write('+(%s)  ' % (ring_buffer[0][0].timestamp,))
                st = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if \
            (num_unvoiced > 0.9 * ring_buffer.maxlen and frame.timestamp + frame.duration - st > minimal_segment_duration) \
            or \
            (frame.timestamp + frame.duration - st > maximal_segment_duration):
                #sys.stdout.write('-(%s)  ' % (frame.timestamp + frame.duration))                
                en = frame.timestamp + frame.duration
                triggered = False
                yield (st,en,en-st,b''.join([f.bytes for f in voiced_frames]))
                ring_buffer.clear()
                voiced_frames = []
        
    if triggered:
        #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        last_frame_en = frame.timestamp + frame.duration
    #sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    
    if voiced_frames:
        yield (st,last_frame_en,last_frame_en-st,b''.join([f.bytes for f in voiced_frames]))


def get_segments(audio, sample_rate, minimal_silence_dur, minimal_segment_duration, maximal_segment_duration):
    vad = webrtcvad.Vad(2)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    return vad_collector(sample_rate, 30, minimal_silence_dur, vad, frames, minimal_segment_duration, maximal_segment_duration)


def read_input(input):
    input_name = input.name
    if input_name.endswith(".wav"):
        input.close()
        yield input_name
        return
    else:
        for input_name in input:
            input_name = input_name.strip()
            if input_name.endswith(".wav"):
                yield input_name
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-op','--output-path', help="Path to output files. Without path splitted audiofiles ARE NOT generated.")
    parser.add_argument('-m', '--minimal-segment-duration', type=int, help="Do not make files shorter then INT seconds", default=5)
    parser.add_argument('-M', '--maximal-segment-duration', type=int, help="Do not make files loneger then INT seconds (forcing cut, overwrites minimal seg. dur.)", default=90)
    parser.add_argument('-s', '--minimal-silence-duration', type=int, help="Do not make files shorter then INT miliseconds", default=150)
    parser.add_argument('-i', '--input', help="Input file(s) file.wav or files.scp or stdin (as list)", type=argparse.FileType(mode='r'), default=(None if sys.stdin.isatty() else sys.stdin))
    parser.add_argument('-u', '--output-utt2spk', help="Output utt2spk, file or -",type=argparse.FileType(mode='w', encoding='utf-8'), default=(None))
    parser.add_argument('-w', '--output-wavscp', help="Output wav.scp with utt and sox split commands, file or -",type=argparse.FileType(mode='w', encoding='utf-8'), default=(None))
    parser.add_argument('-sx', '--output-sox', help="Output with sox split commands, file or -",type=argparse.FileType(mode='w', encoding='utf-8'), default=(None))
    parser.add_argument('-br', '--brownnoise', help="Mix output with brown noise at vol ",type=float, default=0.001)
    parser.add_argument('-v', '--verbose', help="Write some info to stderr", action='store_true')
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit("\nPlease provide path to input file or pipe it via stdin\n")


    for wav_file in read_input(args.input):
        if args.verbose:
            sys.stderr.write(f'Processing {wav_file}...')
        audio, sample_rate = read_wave(wav_file)
        for i, seg in enumerate(get_segments(audio, sample_rate, args.minimal_silence_duration, args.minimal_segment_duration, args.maximal_segment_duration)):            
            st, en, dur, segment = seg
            
            wav_utt_name = f'{os.path.basename(wav_file)[:-4]}-part{i:05}_st{st:.02f}_dur{dur:.02f}'
            
            wav_output_path = os.path.join(args.output_path or '',wav_utt_name+".wav")

            spk_name = f'{os.path.basename(wav_file)[:-4]}'

            utt2spk_line = f'{spk_name} {wav_utt_name}'
            
            wavscp_line = f'{wav_utt_name} sox -V1 {wav_file} -p synth brownnoise vol {args.brownnoise} |  sox -V1 -m - {wav_file} -r 16000 -c1 -b16 -t wavpcm -  trim {st:.02f} {dur:.02f} |'
            
            sox_line = f'sox -V1 {wav_file} {wav_output_path} trim {st:.02f} {dur:.02f}'



            if args.output_path:
                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path, exist_ok=True)
                write_wave(wav_output_path, segment, sample_rate)
            
            if args.output_sox:
                args.output_sox.write(f'{sox_line}\n')

            if args.output_wavscp:
                args.output_wavscp.write(f'{wavscp_line}\n')

            if args.output_utt2spk:
                args.output_utt2spk.write(f'{utt2spk_line}\n')

        if args.verbose:
            sys.stderr.write(f' {i+1} segments created.\n')
