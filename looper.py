# !/usr/bin/env python3
"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""

import sys
import jack
import struct
import json
import numpy as np
import re
import tty
import select
import time
import os
from bisect import bisect_left, insort
from sty import fg, rs, ef
import argparse

parser = argparse.ArgumentParser(description='Audio looper.')
parser.add_argument('project', type=str, metavar='PROJECT', nargs='?', default='unnamed',
                    help='name of the project')

args = parser.parse_args()

PROJECT = args.project

K_ESC = chr(27)
K_ENTER = chr(10)
K_TAB = chr(9)
K_BACKSPACE = chr(127)
K_DELETE = chr(126)

C_MUTE = fg.yellow + ef.bold + '⊘' + rs.all
C_REC_PREP = fg.red + ef.bold + '◉' + rs.all
C_RECORDING = fg.red + ef.blink + ef.bold + '◉' + rs.all
C_MULTI_REC = fg.li_blue + ef.bold + '◉' + rs.all
C_PLAY = fg.li_green + ef.bold + ef.blink + '▶' + rs.all
C_STOP = fg.da_white + ef.bold + '▣' + rs.all
C_METRONOME = fg.li_green + ef.bold + '◉' + rs.all
C_PATTERN = fg.li_green + ef.bold + 'P' + rs.all
C_SONG = fg.li_blue + ef.bold + 'S' + rs.all
C_Q_HARD = fg.li_green + ef.bold + 'ON' + rs.all
C_Q_SOFT = fg.yellow + ef.bold + 'on' + rs.all
C_Q_OFF = fg.da_white + ef.bold + 'off' + rs.all

Q_HARD = 2
Q_SOFT = 1
Q_OFF = 0

MIDI_NOTEON = 0x9
MIDI_NOTEOFF = 0x8

jack_client = None
play_object = None

# Presist some song_ui data
song_ui_mode = 'pattern_edit'
song_insert_pos = 0

# @profile
def jack_process_callback(frames):
    assert frames == jack_client.blocksize
    transport_state, transport_data = jack_client.client.transport_query()
    current_frame = transport_data['frame']

    if isinstance(play_object, Pattern):
        for i, c in enumerate(play_object.channels):
            data = jack_client.in_ports[i].get_array()
            midi_events = jack_client.midi_in_ports[i].incoming_midi_events()

            # If not rolling, just passtrough
            if transport_state != jack.ROLLING:
                jack_client.out_ports[i].get_array()[:] = data
                jack_client.midi_out_ports[i].clear_buffer()
                for offset, event in midi_events:
                    jack_client.midi_out_ports[i].write_midi_event(offset, event)
                continue

            jack_client.out_ports[i].get_array()[:] = c.process(current_frame, data) + data
        
            midi_output = c.process_midi(current_frame, frames, midi_events)
            jack_client.midi_out_ports[i].clear_buffer()
            if midi_output:
                for offset, event in midi_output:
                    try:
                        jack_client.midi_out_ports[i].write_midi_event(offset, event)
                    except Exception as e:
                        print(offset, event, e)

            jack_client.metronome_port.get_array()[:] = (
                play_object._metronome_channel.process(current_frame, data)
            )


    elif isinstance(play_object, Song) and transport_state == jack.ROLLING:
        channels, midi_channels = play_object.process(current_frame, frames)
        for i, cdata in enumerate(channels):
            jack_client.out_ports[i].get_array()[:] = cdata

        for i, midi_output in enumerate(midi_channels):
            jack_client.midi_out_ports[i].clear_buffer()
            if midi_output:
                for offset, event in midi_output:
                    try:
                        jack_client.midi_out_ports[i].write_midi_event(offset, event)
                    except Exception as e:
                        print(offset, event, e)


def jack_shutdown_callback(status, reason):
    global jack_client
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)
    jack_client.connected = False


class JackClient(object):
    def __init__(self):
        self.client = jack.Client('LoopTime')
        self.samplerate = self.client.samplerate
        self.blocksize = self.client.blocksize
        self.silence = np.zeros(self.blocksize, dtype=np.float32)

        # Audio Ports
        self.in_ports = [
            self.client.inports.register(f'channel_{i+1}_in')
            for i in range(8)
        ]
        self.out_ports = [
            self.client.outports.register(f'channel_{i+1}_out')
            for i in range(8)
        ]
        self.metronome_port = self.client.outports.register('metronome')

        # MIDI Ports
        self.midi_in_ports = [
            self.client.midi_inports.register(f'channel_{i+1}_midi_in')
            for i in range(8)
        ]
        self.midi_out_ports = [
            self.client.midi_outports.register(f'channel_{i+1}_midi_out')
            for i in range(8)
        ]

        self.client.set_process_callback(jack_process_callback)
        self.client.set_shutdown_callback(jack_shutdown_callback)
        self.connected = True

    def stop(self):
        self.client.transport_stop()
        self.metronome_port.get_array()[:] = jack_client.silence
        for i in range(8):
            self.out_ports[i].get_array()[:] = jack_client.silence
            self.midi_out_ports[i].clear_buffer()


class Channel(object):
    def __init__(self, buf_size):
        self.buf_size = buf_size
        self._rec = False
        self.mute = False
        self.midi_del = False
        self._vol = 100
        self._fvol = 1.0
        self.clear()

    def clear(self):
        self.buf = np.zeros(self.buf_size, dtype=np.float32)
        self._buf_vol = self.buf * self._fvol
        self._undo_buf = np.copy(self.buf)

        # MIDI data is stored in midi_buf and midi_pos has the frame, which
        # makes lookups faster
        self.midi_buf = []
        self.midi_pos = []
        self._undo_midi_buf = []
        self._undo_midi_pos = []
        self._midi_in_buf = {}

    @property
    def rec(self):
        return self._rec

    @rec.setter
    def rec(self, value):
        if value == self._rec:
            return

        self.save()
        self._rec = value

        self._midi_in_buf = {}

    def save(self):
        self._undo_buf = np.copy(self.buf)
        self._undo_midi_buf = list(self.midi_buf)
        self._undo_midi_pos = list(self.midi_pos)

    def undo(self):
        self.buf = np.copy(self._undo_buf)
        self._buf_vol = self.buf * self._fvol
        self.midi_buf = list(self._undo_midi_buf)
        self.midi_pos = list(self._undo_midi_pos)

    @property
    def vol(self):
        return self._vol

    @vol.setter
    def vol(self, value):
        self._vol = max(0, min(200, value))
        self._fvol = self._vol / 100
        self._buf_vol = self.buf * self._fvol

    def resize(self, buf_size):
        old_size = self.buf_size
        self.buf_size = buf_size
        self.buf = np.resize(self.buf, buf_size)
        self._buf_vol = self.buf * self._fvol

        if self.midi_buf:
            if old_size < buf_size:
                # If new size is bigger, copy items to fill new space
                items_to_copy = True
                while items_to_copy:
                    old_midi_buf = self.midi_buf.copy()
                    old_midi_pos = self.midi_pos.copy()
                    for offset, event in zip(old_midi_pos, old_midi_buf):
                        new_offset = offset + old_size
                        if new_offset < buf_size:
                            self.midi_buf.append(event)
                            self.midi_pos.append(new_offset)
                            items_to_copy = False
            elif old_size > buf_size:
                # If new size is smaller, remove the extra ones
                tmp_midi_buf, tmp_midi_pos = [], []
                for offset, event in zip(self.midi_pos, self.midi_buf):
                    if offset >= buf_size:
                        continue
                    tmp_midi_buf.append(event)
                    tmp_midi_pos.append(offset)
                self.midi_buf = tmp_midi_buf
                self.midi_pos = tmp_midi_pos

        self.save()

    def process(self, pos, data):
        # if muted just return zeroes
        if self.mute:
            return jack_client.silence

        ldata = jack_client.blocksize
        lbuf = self.buf_size
        pos = pos % self.buf_size
        pos_to = (pos + ldata) % lbuf
        if pos_to > pos:
            output = np.copy(self._buf_vol[pos:pos_to])
            if self.rec:
                self.buf[pos:pos_to] += data
                self._buf_vol[pos:pos_to] += data * self._fvol
        else:
            output = np.concatenate((self._buf_vol[pos:], self._buf_vol[:pos_to]))
            if self.rec:
                self.buf[pos:] += data[:ldata-pos_to]
                self.buf[:pos_to] += data[ldata-pos_to:]
                self.buf[pos:] += data[:ldata-pos_to] * self._fvol
                self.buf[:pos_to] += data[ldata-pos_to:] * self._fvol

        return output

    def process_midi(self, pos, frames, events):
        if self.mute:
            return []

        # Find the position of first and last items
        lframes = self.buf_size
        output = [] 

        lbuf = len(self.midi_buf)

        # Add new events to the output, and to the buffer ir recording
        for offset, indata in events:
            if len(indata) == 3:
                status, pitch, vel = struct.unpack('3B', indata)

                # Pass data through
                if not self.midi_del:
                    insort(output, (offset, (status, pitch, vel)))

                if not self.rec:
                    continue

                if status >> 4 == MIDI_NOTEON:
                    if self.midi_del:
                        i = 0
                        while i < len(self.midi_buf):
                            _, tmp_pitch, _ = self.midi_buf[i]
                            if tmp_pitch == pitch:
                                self.midi_pos.pop(i)
                                self.midi_buf.pop(i)
                            else:
                                i += 1
                        lbuf = len(self.midi_buf)
                    else:
                        self._midi_in_buf[pitch] = (offset + pos, status, vel)
                elif status >> 4 == MIDI_NOTEOFF:
                    start_offset, start_status, start_vel = self._midi_in_buf.pop(pitch, [None, None, None])
                    if start_offset is None:
                        continue

                    rel_offset = start_offset % lframes

                    if play_object.quantize == Q_HARD:
                        rel_offset = (
                            round(rel_offset / play_object._quantize) * play_object._quantize
                        ) % lframes
                        # Prevent first beat to end up last by rounding error
                        if lframes - rel_offset < play_object._quantize:
                            rel_offset = 0
                    elif play_object.quantize == Q_SOFT:
                        rel_offset = ((
                            round(rel_offset / play_object._quantize) * play_object._quantize +
                            rel_offset
                        ) // 2) % lframes
                        if lframes - rel_offset < play_object._quantize:
                            rel_offset = 0

                    # Try to find a previous instance of the same note on for
                    # that pitch and replace its velocity. Useful for drums
                    i = bisect_left(self.midi_pos, rel_offset)

                    j = 0
                    new_note = True
                    while j < lbuf:
                        p = (j + i) % lbuf
                        if self.midi_pos[p] != rel_offset:
                            break
                        tmp_status, tmp_pitch, tmp_vel = self.midi_buf[p]
                        if tmp_status >> 4 == MIDI_NOTEON and pitch == tmp_pitch:
                            new_note = False
                            # We found the note. Update the velocity
                            self.midi_buf[p] = (tmp_status, tmp_pitch, start_vel)
                            break
                        j += 1

                    if new_note:
                        # Totally new note, let's insert
                        self.midi_pos.insert(i, rel_offset)
                        self.midi_buf.insert(i, (start_status, pitch, start_vel))

                        # Insert note off
                        note_off_offset = (rel_offset + (offset + pos - start_offset)) % lframes
                        i = bisect_left(self.midi_pos, note_off_offset)
                        self.midi_pos.insert(i, note_off_offset)
                        self.midi_buf.insert(i, (status, pitch, vel))

        # If there is any item to play

        pos_buf_start = pos - (pos % lframes)
        pos_start = pos - pos_buf_start
        pos_to = pos_start + frames

        #find the best place to start looping
        i = bisect_left(self.midi_pos, pos % lframes)
        j = 0
        while j < lbuf:
            p = (j + i) % lbuf
            frame = self.midi_pos[p]

            if frame < pos_start:
                frame += lframes

            if frame >= pos_to:
                break

            insort(output, (frame - pos_start, tuple(self.midi_buf[p])))
            j += 1

        return output
                

class Metronome(Channel):
    def __init__(self, buf_size, bars, clicks):
        self.buf_size = buf_size
        bars = bars * clicks

        part_size = buf_size // bars

        parts = []
        for i in range(bars):
            freq = 1000 if i else 1200
            parts.append(self.wave(freq, part_size))

        if buf_size - part_size * bars:
            parts.append(np.zeros(buf_size - part_size * bars, dtype=np.float32))

        self.buf = np.concatenate(parts)
        self._rec = False
        self.mute = False
        self.vol = 100

    def wave(self, freq, buf_size):
        samples = int(jack_client.samplerate*0.02)
        t = np.arange(samples)/jack_client.samplerate
        return np.concatenate((
            np.sin(2 * np.pi * freq * t, dtype=np.float32),
            np.zeros(buf_size - samples, dtype=np.float32)
        )) * 0.2

    def process(self, pos, data):
        ldata = jack_client.blocksize
        
        # Short circuit if muted
        if self.mute:
            return np.zeros(ldata, dtype=np.float32)

        pos = pos % self.buf_size
        lbuf = self.buf_size
        pos_to = (pos + ldata) % lbuf

        if pos_to > pos:
            # To support volume we should multiply by self._vol / 100. Not
            # doing to get some extra cycles
            return self.buf[pos:pos_to]

        # To support volume we should multiply by self._vol / 100. Not
        # doing to get some extra cycles
        return np.concatenate((self.buf[pos:], self.buf[:pos_to]))


class Pattern(object):
    def __init__(self, bpm=90, bars=4, name='unnamed', savefile=None):
        self.multi_rec = False
        self.midi_del = False
        self.name = name
        self.bpm = bpm
        self.bars = bars
        self.channels = [
            Channel(self.buffer_size())
            for _ in range(8)
        ]
        self.savefile = savefile
        self.load()
        self._metronome = True
        self.metronome_click = 1
        self.quantize = Q_OFF
        self.quantize_click = 1

    @property
    def bpm(self):
        return self._bpm

    @bpm.setter
    def bpm(self, value):
        self._bpm = max(45, min(240, value))

    @property
    def bars(self):
        return self._bars

    @bars.setter
    def bars(self, value):
        self._bars = max(1, min(64, value))

    @property
    def metronome(self):
        return self._metronome

    @metronome.setter
    def metronome(self, value):
        self._metronome = value
        self._metronome_channel.mute = not self._metronome

    @property
    def metronome_click(self):
        return self._metronome_click

    @metronome_click.setter
    def metronome_click(self, value):
        self._metronome_click = max(1, min(9, value))
        self._set_metronome()

    @property
    def quantize_click(self):
        return self._quantize_click

    @quantize_click.setter
    def quantize_click(self, value):
        self._quantize_click = max(1, min(16, value))
        self._set_quantize()

    def _set_quantize(self):
        self._quantize = int(self.buffer_size() / self.bars / self._quantize_click)

    def _set_metronome(self):
        self._metronome_channel = Metronome(
            self.buffer_size(),
            self.bars,
            self._metronome_click
        )
        self._metronome_channel.mute = not self._metronome

    def buffer_size(self):
        buf_time = 60/self.bpm*self.bars
        return int(buf_time * jack_client.samplerate)

    def resize(self):
        buf_size = self.buffer_size()
        for c in self.channels:
            c.resize(buf_size)
        self._set_metronome()
        self._set_quantize()

    def dump(self):
        if self.savefile is None:
            return

        if not os.path.exists(self.savefile):
            os.makedirs(self.savefile)

        np.savez_compressed(
            f'{self.savefile}/pattern_{self.name}_channels_data.npz',
            [c.buf for c in self.channels]
        )
        with open(f'{self.savefile}/pattern_{self.name}_meta', 'w') as f:
            channel_midi = []
            for c in self.channels:
                cdata = []
                for data in zip(c.midi_pos, c.midi_buf):
                    cdata.append(data)
                channel_midi.append(cdata)
                
            f.write(json.dumps({
                'bpm': self.bpm,
                'bars': self.bars,
                'channel_volumes': [c.vol for c in self.channels],
                'channel_midi': channel_midi,
            }))

    def load(self):
        filename = f'{self.savefile}/pattern_{self.name}_meta'
        if self.savefile is None or not os.path.exists(filename):
            return

        metadata = json.loads(open(f'{self.savefile}/pattern_{self.name}_meta').read())
        self.bpm = metadata['bpm']
        self.bars = metadata['bars']
        buf_size = self.buffer_size()
        channels_data = np.load(
            f'{self.savefile}/pattern_{self.name}_channels_data.npz'
        )
        for c, data, vol, channel_midi in zip(self.channels, channels_data['arr_0'], metadata['channel_volumes'], metadata.get('channel_midi', [])):
            c.buf = data
            c.buf_size = buf_size
            c.vol = vol
            c.midi_buf = [b for p, b in channel_midi]
            c.midi_pos = [p for p, b in channel_midi]
            c.save()


class Song(object):
    def __init__(self, bpm=90, savefile=None):
        self.bpm = bpm
        self.patterns = {}
        self.playlist = []
        self.savefile = savefile
        self.load()

    @property
    def bpm(self):
        return self._bpm

    @bpm.setter
    def bpm(self, value):
        self._bpm = max(45, min(240, value))

    def dump(self):
        if self.savefile is None:
            return

        if not os.path.exists(self.savefile):
            os.makedirs(self.savefile)

        with open(f'{PROJECT}/song_meta', 'w') as f:
            f.write(json.dumps({
                'bpm': self.bpm,
                'patterns': list(self.patterns.keys()),
                'playlist': [p.name for _, _, p in self.playlist],
            }))

        for pattern in self.patterns.values():
            pattern.dump()

    def load(self):
        if self.savefile is None or not os.path.exists(self.savefile):
            return

        metadata = json.loads(open(f'{self.savefile}/song_meta').read())
        self.bpm = metadata['bpm']
        for pattern_name in metadata['patterns']:
            self.patterns[pattern_name] = Pattern(
                name=pattern_name,
                savefile=self.savefile,
            )
            self.patterns[pattern_name].load()
        for p_name in metadata['playlist']:
            self.insert_pattern(len(self.playlist), self.patterns[p_name])

    def insert_pattern(self, pos, pattern):
        self.playlist.insert(pos, [None, None, pattern])
        self.recalculate_playlist()

    def delete_pattern(self, pattern_name):
        del self.patterns[pattern_name]
        i = 0
        while i < len(self.playlist):
            _, _, pattern = self.playlist[i]
            if pattern.name != pattern_name:
                i += 1
            else:
                self.playlist.pop(i)
        self.recalculate_playlist()

    def delete_playlist_item(self, pos):
        if pos < len(self.playlist):
            self.playlist.pop(pos)
            self.recalculate_playlist()

    def recalculate_playlist(self):
        # Recalculate frame locations
        start = 0
        for item in self.playlist:
            end = start + item[2].buffer_size()
            item[0] = end
            item[1] = start
            start = end

    def process(self, pos, frames):
        starting_pattern = bisect_left(self.playlist, [pos])
        i = starting_pattern

        if i >= len(self.playlist):
            return [], [[] for _ in range(8)]

        frame_end, frame_start, play_pattern = self.playlist[i]

        channel_pos = pos - frame_start
        channels = []
        midi_channels = []

        for c in range(8):
            channels.append(play_pattern.channels[c]._buf_vol[channel_pos:channel_pos+frames])
            remaining_frames = frames - channels[c].size
            # If not enough frames, try to complete with the next pattern in list
            if remaining_frames:
                if i+1 < len(self.playlist):
                    frame_end, frame_start, play_pattern = self.playlist[i+1]
                    channels[c] = np.concatenate((
                        channels[c],
                        play_pattern.channels[c]._buf_vol[:remaining_frames],
                    ))
            # If still remaining frames, then complete with zeroes
            remaining_frames = frames - channels[c].size
            if remaining_frames:
                channels[c] = np.concatenate((
                    channels[c],
                    np.zeros(remaining_frames, dtype=np.float32),
                ))

            # MIDI Play       
            output = [] 

            add_output = True
            pattern_i = starting_pattern
            while add_output and pattern_i < len(self.playlist):
                frame_end_midi, frame_start_midi, play_pattern_midi = self.playlist[pattern_i]
                midi_buf = play_pattern_midi.channels[c].midi_buf
                midi_pos = play_pattern_midi.channels[c].midi_pos

                lbuf = len(midi_buf)
                pos_start = pos - frame_start_midi
                pos_to = pos_start + frames
                first_item = bisect_left(midi_pos, pos_start)

                # If there is any item to play
                if first_item != lbuf:
                    i = first_item
                    while i < lbuf:
                        frame = midi_pos[i]
                        if frame >= pos_to:
                            # Reached end of frames
                            add_output = False
                            break

                        insort(output, (frame - pos_start, midi_buf[i]))
                        i += 1
                pattern_i += 1

            midi_channels.append(output)

        return channels, midi_channels


def print_pattern(pattern, playing):
    # Reset position
    sys.stdout.write('\033[1;1H')

    data = []

    data.append(pattern.name.upper())

    data.append(C_MULTI_REC if pattern.multi_rec else ' ')
    data.append(
        (C_RECORDING if playing else C_REC_PREP) if pattern.midi_del else ' '
    )

    for c in pattern.channels:
        data.append(
            (C_RECORDING if playing else C_REC_PREP) if c.rec else ' '
        )

    data.append(f'{pattern.bpm:3}')
    data.append(f'{pattern.bars:2}')

    for c in pattern.channels:
        data.append(C_MUTE if c.mute else ' ')

    data.append(C_PLAY if playing else C_STOP)

    for c in pattern.channels:
        data.append(f'{c.vol:3}')

    data.append(C_METRONOME if pattern.metronome else ' ')
    data.append(f'{pattern.metronome_click}')

    data.append(
        C_Q_HARD if pattern.quantize == Q_HARD else
        C_Q_SOFT if pattern.quantize == Q_SOFT else
        C_Q_OFF
    )
    data.append(f'{pattern.quantize_click}')

    template = (
        '                      EDIT PATTERN *                            \n'
        'Channel   1   2   3   4   5   6   7   8     Mult [*] Note Del[*]\n'
        '        ┏━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━┓  ┏━━━━━━━━━━━┳━━━┳━━━┓\n'
        '    Rec ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃  ┃ BPM/Bars  ┃*  ┃ * ┃\n'
        '        ┠───╂───╂───╂───╂───╂───╂───╂───┨  ┠───────────╂───╊━━━┛\n'
        '   Mute ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃  ┃ Transport ┃ * ┃    \n'
        '        ┠───╂───╂───╂───╂───╂───╂───╂───┨  ┠───────────╂───╊━━━┓\n'
        '    Vol ┃*  ┃*  ┃*  ┃*  ┃*  ┃*  ┃*  ┃*  ┃  ┃ Metronome ┃ * ┃ * ┃\n'
        '        ┗━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┛  ┠───────────╂───╂───┨\n'
        '                                           ┃ MIDI Q/IN ┃*  ┃ * ┃\n'
        '                                           ┗━━━━━━━━━━━┻━━━┻━━━┛\n'
        '                                                                \n'
        '                                                                \n'
        '                                                                \n'
        '\033[6A'
    )

    for d in data:
        pos = template.find('*')
        if pos == -1:
            raise Exception('More data items than markers in template')
        len_d = len(re.sub('\033.*?m', '', d))
        template = template[:pos] + d + template[pos+len_d:]

    print(template)


def pattern_ui(pattern):
    global play_object
    play_object = pattern
    rec_keys = ['1', '2', '3', '4', '5', '6', '7', '8']
    mute_keys = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i']
    vol_up_keys = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k']
    vol_down_keys = ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',']

    while jack_client.connected:
        
        playing = jack_client.client.transport_state == jack.ROLLING
        print_pattern(pattern, playing)

        if select.select([sys.stdin], [], [], 0.5) != ([sys.stdin], [], []):
            continue

        c = sys.stdin.read(1).lower()

        if c in rec_keys:
            channel = pattern.channels[rec_keys.index(c)]
            if not pattern.multi_rec:
                for c in pattern.channels:
                    if c != channel:
                        c.rec = False
            channel.rec = not channel.rec
        elif c in mute_keys:
            channel = pattern.channels[mute_keys.index(c)]
            channel.mute = not channel.mute
        elif c in vol_up_keys:
            channel = pattern.channels[vol_up_keys.index(c)]
            channel.vol += 1
        elif c in vol_down_keys:
            channel = pattern.channels[vol_down_keys.index(c)]
            channel.vol += -1
        elif c == '9':
            pattern.multi_rec = not pattern.multi_rec
            if not pattern.multi_rec:
                for c in pattern.channels:
                    c.rec = False
        elif c == '0':
            pattern.midi_del = not pattern.midi_del
            for c in pattern.channels:
                c.midi_del = pattern.midi_del
        elif c == '\\':
            pattern.metronome = not pattern.metronome
        elif c == ']':
            pattern.metronome_click += 1
        elif c == '[':
            pattern.metronome_click += -1
        elif c == '/':
            pattern.quantize = (
                Q_HARD if pattern.quantize == Q_OFF else
                Q_SOFT if pattern.quantize == Q_HARD else
                Q_OFF
            )
        elif c == '>':
            pattern.quantize_click += 1
        elif c == '<':
            pattern.quantize_click += -1
        elif c == ' ':
            if playing:
                jack_client.stop()
            else:
                jack_client.client.transport_start()
        elif c == K_ENTER:
            for c in pattern.channels:
                c.save()
        elif c == K_ESC:
            for c in pattern.channels:
                c.undo()
        elif c == K_BACKSPACE:
            print('DELETE CHANNEL CONTENT - INPUT CHANNEL')
            c = sys.stdin.read(1).lower()
            if c not in rec_keys:
                print('CANCELLED')
                time.sleep(1.5)
                continue

            print(f'CONFIRM DELETE CHANNEL {c} Y/N')
            yn = sys.stdin.read(1).lower()
            if yn != 'y':
                print('CANCELLED')
                time.sleep(1.5)
                continue

            channel = pattern.channels[rec_keys.index(c)]
            channel.clear()
        elif c == '=':
            print("CHANGE BPM/BARS")
            sys.stdout.write(f'INPUT BPM (ESCAPE TO KEEP {pattern.bpm}) ')
            c = None
            tmp_bpm = ''
            while c != K_ESC:
                c = sys.stdin.read(1).lower()
                if c == K_ENTER:
                    break
                elif '0' <= c <= '9':
                    tmp_bpm += c
                    sys.stdout.write(c)
                elif c == K_ESC:
                    tmp_bpm = ''
            tmp_bpm = tmp_bpm or pattern.bpm

            sys.stdout.write(f'\nINPUT BARS (ESCAPE TO KEEP {pattern.bars}) ')
            c = None
            tmp_bars = ''
            while c != K_ESC:
                c = sys.stdin.read(1).lower()
                if c == K_ENTER:
                    break
                elif '0' <= c <= '9':
                    tmp_bars += c
                    sys.stdout.write(c)
                elif c == K_ESC:
                    tmp_bars = ''
            tmp_bars = tmp_bars or pattern.bars

            sys.stdout.write(f'\nCONFIRM BPM:{tmp_bpm} BARS:{tmp_bars} Y/N')
            yn = sys.stdin.read(1).lower()
            if yn != 'y':
                sys.stdout.write('\nCANCELLED')
                time.sleep(1.5)
                continue

            pattern.bpm = int(tmp_bpm)
            pattern.bars = int(tmp_bars)
            pattern.resize()

        elif c == K_TAB:
            break


def print_song(song, playing, edit_mode, insert_pos, no_playlist=False):
    # Reset position
    sys.stdout.write('\033[1;1H')

    pattern_rows = [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0',],
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm',],
    ]

    data = []

    for pattern_name in pattern_rows[0]:
        label = pattern_name.upper()
        if pattern_name in song.patterns:
            label = fg.yellow + ef.bold + label + rs.all
        data.append(label)

    data.append(f'{song.bpm:3}')

    for pattern_name in pattern_rows[1]:
        label = pattern_name.upper()
        if pattern_name in song.patterns:
            label = fg.yellow + ef.bold + label + rs.all
        data.append(label)

    data.append(C_PLAY if playing else C_STOP)

    for pattern_name in pattern_rows[2]:
        label = pattern_name.upper()
        if pattern_name in song.patterns:
            label = fg.yellow + ef.bold + label + rs.all
        data.append(label)

    data.append(C_PATTERN if edit_mode == 'pattern_edit' else C_SONG)

    for pattern_name in pattern_rows[3]:
        label = pattern_name.upper()
        if pattern_name in song.patterns:
            label = fg.yellow + ef.bold + label + rs.all
        data.append(label)

    template = (
        '              PATTERNS                                          \n'
        ' ┏━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━┳━━━┓ ┏━━━━━━━━━━━┳━━━┓    \n'
        ' ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ ┃    BPM    ┃*  ┃    \n'
        ' ┠───╂───╂───╂───╂───╂───╂───╂───╂───╂───┨ ┠───────────╂───┨    \n'
        ' ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ ┃ Transport ┃ * ┃    \n'
        ' ┠───╂───╂───╂───╂───╂───╂───╂───╂───╊━━━┛ ┠───────────╂───┨    \n'
        ' ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃     ┃ Edit Mode ┃ * ┃    \n'
        ' ┠───╂───╂───╂───╂───╂───╂───╊━━━┻━━━┛     ┗━━━━━━━━━━━┻━━━┛    \n'
        ' ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃ * ┃                                  \n'
        ' ┗━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┛                                  \n'
        '                                                                \n'
        '                                                                \n'
        '                                                                \n'
        '                                                                \n'
        '                                                                \n'
        '\033[6A'
    )

    for d in data:
        pos = template.find('*')
        if pos == -1:
            raise Exception('More data items than markers in template')
        len_d = len(re.sub('\033.*?m', '', d))
        template = template[:pos] + d + template[pos+len_d:]

    print(template)

    if no_playlist:
        return

    if edit_mode == 'pattern_edit':
        insert_pos = -1

    playlist = []
    arr_playlist = []
    for i, p in enumerate(song.playlist):
        if playing and p[0] >= jack_client.client.transport_frame >= p[1]:
            playlist.append(fg.green + ef.bold + ef.blink + p[2].name.upper() + rs.all)
        elif i == insert_pos:
            playlist.append(fg.yellow + ef.bold + p[2].name.upper() + rs.all)
        else:
            playlist.append(p[2].name.upper())
        if len(playlist) == 30:
            arr_playlist.append(' '.join(playlist))
            playlist = []
    
    if len(playlist):
        arr_playlist.append(' '.join(playlist))

    for line in arr_playlist:
        print(' ' + line)


def song_ui(song):
    global play_object
    global song_ui_mode
    global song_insert_pos

    play_object = song
    pattern_keys = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
        'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
        'z', 'x', 'c', 'v', 'b', 'n', 'm',
    ]

    while jack_client.connected:
        playing = jack_client.client.transport_state == jack.ROLLING
        print_song(song, playing, song_ui_mode, song_insert_pos)

        if select.select([sys.stdin], [], [], 0.5) != ([sys.stdin], [], []):
            continue

        c = sys.stdin.read(1).lower()

        if c in pattern_keys:
            if song_ui_mode == 'pattern_edit' or not c in song.patterns:
                if not c in song.patterns:
                    song.patterns[c] = Pattern(
                        bpm=song.bpm,
                        name=c,
                        savefile=song.savefile,
                    )
                return song.patterns[c]
            elif c in song.patterns:
                song.insert_pattern(song_insert_pos+1, song.patterns[c])
                song_insert_pos = min(song_insert_pos + 1, len(song.playlist)-1)
        elif c == ']' and song_ui_mode == 'song_edit':
            song_insert_pos = min(len(song.playlist)-1, song_insert_pos + 1)
            if not playing:
                jack_client.client.transport_frame = song.playlist[song_insert_pos][1]
        elif c == '[' and song_ui_mode == 'song_edit':
            song_insert_pos = max(0, song_insert_pos - 1)
            if not playing:
                jack_client.client.transport_frame = song.playlist[song_insert_pos][1]

        elif c == ' ':
            if playing:
                jack_client.stop()
            else:
                jack_client.client.transport_start()
        elif c == K_BACKSPACE:
            if song_ui_mode == 'pattern_edit':
                print_song(song, playing, song_ui_mode, song_insert_pos, no_playlist=True)
                
                print('DELETE PATTERN - INPUT PATTERN')
                p = sys.stdin.read(1).lower()
                if p not in song.patterns:
                    print('CANCELLED')
                    time.sleep(1.5)
                    continue

                print(f'CONFIRM DELETE PATTERN {c} Y/N')
                yn = sys.stdin.read(1).lower()
                if yn != 'y':
                    print('CANCELLED')
                    time.sleep(1.5)
                    continue

                song.delete_pattern(p)

            else:
                song.delete_playlist_item(song_insert_pos)
                song_insert_pos = max(0, min(song_insert_pos, len(song.playlist)-1))
        elif c == '-':
            print_song(song, playing, song_ui_mode, song_insert_pos, no_playlist=True)
            
            sys.stdout.write('COPY PATTERN - INPUT SOURCE PATTERN: ')
            src_p = sys.stdin.read(1).lower()
            print(src_p.upper())
            if src_p not in song.patterns:
                print('CANCELLED')
                time.sleep(1.5)
                continue

            sys.stdout.write('COPY PATTERN - INPUT DESTINATION PATTERN: ')
            dst_p = sys.stdin.read(1).lower()
            print(dst_p.upper())
            copy_mode = 'o'
            if dst_p in song.patterns:
                sys.stdout.write(
                    'DESTINATION EXISTS - '
                    f'{ef.bold+fg.yellow}O{rs.all}VERWRITE/'
                    f'{ef.bold+fg.yellow}S{rs.all}WITCH/'
                    f'{ef.bold+fg.yellow}C{rs.all}ANCEL: '
                )
                copy_mode = sys.stdin.read(1).lower()
                print(copy_mode.upper())
                if copy_mode not in ['o', 's']:
                    print('CANCELLED')
                    time.sleep(1.5)
                    continue

            if copy_mode == 's':
                # Switch pattern objects, change names, and then fix the playlist
                song.patterns[src_p], song.patterns[dst_p] = song.patterns[dst_p], song.patterns[src_p]
                song.patterns[src_p].name, song.patterns[dst_p].name = src_p, dst_p
                for i, data in enumerate(song.playlist):
                    if data[2] == song.patterns[src_p]:
                        song.playlist[i] = [None, None, song.patterns[dst_p]]
                    elif data[2] == song.patterns[dst_p]:
                        song.playlist[i] = [None, None, song.patterns[src_p]]
                song.recalculate_playlist()
            else:
                # The overwrite is a hack, but should be fine
                tmp_dst_pattern = song.patterns.get(dst_p)
                song.patterns[src_p].name = dst_p
                song.patterns[src_p].dump()
                song.patterns[src_p].name = src_p
                song.patterns[dst_p] = Pattern(
                    bpm=song.bpm,
                    name=dst_p,
                    savefile=song.savefile,
                )
                for i, data in enumerate(song.playlist):
                    if data[2] == tmp_dst_pattern:
                        song.playlist[i] = [None, None, song.patterns[dst_p]]
                song.recalculate_playlist()

        elif c == K_ENTER:
            song_ui_mode = 'song_edit' if song_ui_mode == 'pattern_edit' else 'pattern_edit'
        elif c == '=':
            print_song(song, playing, song_ui_mode, song_insert_pos, no_playlist=True)
            print("CHANGE DEFAULT BPM")
            sys.stdout.write(f'INPUT DEFAULT BPM (ESCAPE TO KEEP {song.bpm}) ')
            c = None
            tmp_bpm = ''
            while c != K_ESC:
                c = sys.stdin.read(1).lower()
                if c == K_ENTER:
                    break
                elif '0' <= c <= '9':
                    tmp_bpm += c
                    sys.stdout.write(c)
                elif c == K_ESC:
                    tmp_bpm = ''
            tmp_bpm = tmp_bpm or song.bpm
            song.bpm = int(tmp_bpm)


if __name__ == '__main__':
    jack_client = JackClient()
    song = Song(savefile=PROJECT)

    tty_save = tty.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    sys.stdout.write('\033c')

    with jack_client.client:
        try:
            while True:
                pattern = song_ui(song)
                pattern_ui(pattern)
                jack_client.stop()
        except KeyboardInterrupt:
            pass

    tty_save = tty.tcsetattr(sys.stdin, tty.TCSAFLUSH, tty_save)
    sys.stdout.write('\033c')
    song.dump()
