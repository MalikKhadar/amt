from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi,midi_to_events
from anticipation.config import *
from anticipation.vocab import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--p', required=False, default=0.98, type=float)
parser.add_argument('-m', '--midi_file_path', required=False, default="", type=str)
parser.add_argument('-s', '--start', required=False, default=0, type=float)
parser.add_argument('-e', '--end', required=False, default=10, type=float)
parser.add_argument('-d', '--destination', required=True, type=str)
parser.add_argument('-r', '--replace', action='store_true')
parser.add_argument('-b', '--big_mode', action='store_true')

args = parser.parse_args()
dest = args.destination
midi_file_path = args.midi_file_path
start = args.start
end = args.end
p = args.p
replace = args.replace
big_mode = args.big_mode

SMALL_MODEL = 'stanford-crfm/music-small-800k'     # faster inference, worse sample quality
MEDIUM_MODEL = 'stanford-crfm/music-medium-800k'   # slower inference, better sample quality                                 

# load an anticipatory music transformer
chosen_model = SMALL_MODEL
if big_mode:
    chosen_model = MEDIUM_MODEL
model = AutoModelForCausalLM.from_pretrained(chosen_model).cuda()

# convert events to midi and save at dest
def midify(tokens):
    mid = events_to_midi(tokens)
    mid.save(dest)

def save_program_events():
    saved_midi_events = []
    return saved_midi_events

# generate new midi in buffer
def generate_midi():
    # Default: generate a new midi file with no history
    segment = []
    # Alternatively, if a midi file was specified, use that as history
    if (midi_file_path != ""):
        segment = midi_to_events(midi_file_path)

    history = ops.clip(segment, 0, start, clip_duration=False)

    # Default: accompany, consider the notes in this range
    anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, start, ops.max_time(segment), clip_duration=False)]
    # Alternatively, do not consider the notes in the edit range, replace instead
    if replace:
        anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, end, ops.max_time(segment), clip_duration=False)]

    inpainted = generate(model, start, end, inputs=history, controls=anticipated, top_p=p)
    midify(ops.combine(inpainted, anticipated))

generate_midi()