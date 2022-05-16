import csv
import json
import numpy as np
from PIL import Image


from .builder import DATASETS
from .custom import CustomDataset


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.
    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise ValueError(fmt.format(e))


def _read_annotations(csv_reader, classes):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            ValueError(f'line {line}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\'')

        if img_file not in result:
            im = Image.open(img_file)
            result[img_file] = {'ann': {'bboxes': [], 'labels': []}, 'width': im.width, 'height': im.height}

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file]['ann']['bboxes'].append([x1, y1, x2, y2])
        result[img_file]['ann']['labels'].append(classes.index(class_name))
    return result


def load_annotations(csv_file, classes_file):
    with open(classes_file, 'r') as f:
        classes = [c.strip() for c in f.readlines()]

    with open(csv_file, 'r', newline='') as f:
        csv_reader = csv.reader(f, delimiter=',')
        anns = _read_annotations(csv_reader, classes)

    return anns


@DATASETS.register_module()
class XViewDataset(CustomDataset):
    CLASSES = ['Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter', 'Passenger Vehicle',
               'Small Car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box',
               'Truck Tractor', 'Trailer', 'Truck w/Flatbed', 'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle',
               'Passenger Car', 'Cargo Car', 'Flat Car','Tank car', 'Locomotive', 'Maritime Vessel','Motorboat',
               'Sailboat', 'Tugboat', 'Barge','Fishing Vessel', 'Ferry', 'Yacht','Container Ship', 'Oil Tanker',
               'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 'Straddle Carrier',
               'Mobile Crane','Dump Truck', 'Haul Truck', 'Scraper/Tractor', 'Front loader/Bulldozer', 'Excavator',
               'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed','Building', 'Aircraft Hangar','Damaged Building',
               'Facility', 'Construction Site', 'Vehicle Lot', 'Helipad','Storage Tank', 'Shipping container lot',
               'Shipping Container', 'Pylon', 'Tower']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            anns = json.load(f)

        data = [{'filename': img_fp, 'width': ann_data['width'], 'height': ann_data['height'],
                 'ann': {'bboxes': np.array(ann_data['ann']['bboxes']),
                         'labels': np.array(ann_data['ann']['labels'])}}
                for img_fp, ann_data in anns.items()]
        return data
