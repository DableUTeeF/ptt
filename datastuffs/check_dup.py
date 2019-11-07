from imagededup.methods import PHash
import os
import json


root = '/media/palm/62C0955EC09538ED/ptt/full_sized'
duplicates = []
for cls in os.listdir(root)[1:]:
    phasher = PHash()

    # Generate encodings for all images in an image directory
    encodings = phasher.encode_images(image_dir=os.path.join(root, cls))

    # Find duplicates using the generated encodings
    duplicate = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=1)
    with open('/home/palm/PycharmProjects/ptt/datastuffs/dups/'+cls+'.json', 'w') as write:
        json.dump([duplicate, encodings], write)
    duplicates.append(duplicate)
