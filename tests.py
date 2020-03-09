from unittest import TestCase
from looper import Channel


class TestMidiResize(TestCase):
    def test_resize_half(self):
        c = Channel(8)
        c.midi_buf = ['a', 'b' , 'c']
        c.midi_pos = [0, 3, 4]

        c.resize(4)
        self.assertEquals(
            c.midi_pos,
            [0, 3]
        )
        self.assertEquals(
            c.midi_buf,
            ['a', 'b']
        )

    def test_resize_double(self):
        c = Channel(8)
        c.midi_buf = ['a', 'b' , 'c']
        c.midi_pos = [0, 1, 6]

        c.resize(16)
        self.assertEquals(
            c.midi_pos,
            [0, 1, 6, 8, 9, 14]
        )
        self.assertEquals(
            c.midi_buf,
            ['a', 'b', 'c', 'a', 'b', 'c']
        )

    def test_resize_halfup(self):
        c = Channel(8)
        c.midi_buf = ['a', 'b' , 'c']
        c.midi_pos = [0, 1, 6]

        c.resize(12)
        self.assertEquals(
            c.midi_pos,
            [0, 1, 6, 8, 9]
        )
        self.assertEquals(
            c.midi_buf,
            ['a', 'b', 'c', 'a', 'b']
        )
