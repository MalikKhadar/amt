from transformers import AutoModelForCausalLM

from anticipation import ops
from anticipation.sample import generate
from anticipation.convert import events_to_midi,midi_to_events
from anticipation.config import *
from anticipation.vocab import *
import argparse
import winsound

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--p', required=False, default=0.98, type=float)
#parser.add_argument('-t', '--delta', required=False, default=5, type=float)
parser.add_argument('-m', '--midi_file_path', required=False, default="", type=str)
parser.add_argument('-s', '--start', required=False, default=0, type=float)
parser.add_argument('-e', '--end', required=False, default=10, type=float)
parser.add_argument('-u', '--use_ticks', action='store_true')
parser.add_argument('-d', '--destination', required=True, type=str)
parser.add_argument('-r', '--replace', action='store_true')
parser.add_argument('-b', '--big_mode', action='store_true')
parser.add_argument('-n', '--notification_file_path', required=False, default = "", type=str)

args = parser.parse_args()
dest = args.destination
midi_file_path = args.midi_file_path
start = args.start
end = args.end
use_seconds = not args.use_ticks
p = args.p
#delta = args.delta
replace = args.replace
big_mode = args.big_mode
notification_file_path = args.notification_file_path

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

# generate new midi in buffer
def generate_midi():
    # These args may change based on use_ticks, so declare global
    global start
    global end

    # Default: generate a new midi file with no history
    segment = []
    # Alternatively, if a midi file was specified, use that as history
    if (midi_file_path != ""):
        segment = midi_to_events(midi_file_path)

    history = ops.clip(segment, 0, start, clip_duration=False, seconds=use_seconds)

    # Default: accompany, consider the notes in this range
    anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, start, ops.max_time(segment, seconds=use_seconds), clip_duration=False, seconds=use_seconds)]
    # Alternatively, do not consider the notes in the edit range, replace instead
    if replace:
        anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, end, ops.max_time(segment, seconds=use_seconds), clip_duration=False, seconds=use_seconds)]
        
    # If using ticks, account for the time resolution that will be multiplied in during generate
    if not use_seconds:
        start /= TIME_RESOLUTION
        end /= TIME_RESOLUTION

    # Generate events and write as midi to file
    inpainted = generate(model, start, end, inputs=history, controls=anticipated, top_p=p)#, delta=delta)
    midify(ops.combine(inpainted, anticipated))

    # If a sound was provided, play it to notify script completion
    if notification_file_path != "":
        winsound.PlaySound(notification_file_path, winsound.SND_FILENAME)

generate_midi()