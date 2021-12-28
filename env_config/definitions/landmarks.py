INCLUDE_UNSEEN = True

NUM_LANDMARKS = 64

LANDMARK_RADII = {
    "ConniferCluster": 75,
    "GiantPalm": 75,
    "House": 125,
    "WaterWell": 75,
    "Mushroom": 75,
    "Gorilla": 75,
    "Windmill": 75,
    "WoodBarrel": 75,
    "LionStatue": 125,
    "TreasureChest": 75,
    "Rock": 75,
    "Stump": 75,
    "Pillar": 75,
    "Tombstone": 75,
    "WoodenChair": 75,
    "FireHydrant": 75,
    "OilDrum": 75,
    "TrafficCone": 75,
    "Bench": 125,
    "StreetLamp": 75,
    "PhoneBox": 75,
    "Anvil": 75,
    "Ladder": 75,
    "Apple": 75,
    "Banana": 75,
    "RedFlowers": 75,
    "YellowFlowers": 75,
    "Cactus": 75,
    "RecycleBin": 75,
    "Boat": 125,
    "Barrel": 75,
    "Barrel2": 75,
    "Beacon": 125,
    "BigHouse": 125,
    "Boletus": 75,
    "Box": 75,
    "BushTree3": 75,
    "BushTree2": 75,
    "BushTree": 75,
    "Coach": 125,
    "Column": 75,
    "Container": 110,
    "Dumpster": 75,
    "GoldCone": 75,
    "House1": 100,
    "House2": 100,
    "Jet": 120,
    "LowPolyTree": 75,
    "LpPine": 75,
    "Palm1": 85,
    "Palm2": 95,
    "Palm3": 100,
    "Pickup": 90,
    "Pumpkin": 75,
    "Soldier": 75,
    "SteelCube": 75,
    "Stone1": 75,
    "Stone2": 75,
    "Stone3": 75,
    "Tank": 100,
    "Tower2": 100,
    "TvTower": 75,
    "Well": 95
}

PORTABLE_LANDMARK_RADII = {
    "Mushroom": 100,
    "TrafficCone": 100,
    "Banana": 100,
    "Gorilla": 100,
    "Boat": 100,
    "Palm1": 100,
    "Palm2": 100,
    "Pumpkin": 100,
    "Box": 100,
    "Stump": 100,
    "Tombstone": 100,
    "Jet": 100,
    "House1": 100,
    "Stone1": 75,
    "Container": 110
}

PORTABLE_LANDMARK_STAGE_NAMES = {
    "Mushroom": "1",
    "TrafficCone": "2",
    "Banana": "3",
    "Gorilla": "4",
    "Boat": "5",
    "Palm1": "6",
    "Palm2": "7",
    "Pumpkin": "8",
    "Box": "9",
    "Stump": "10",
    "Tombstone": "11",
    "Jet": "12",
    "House1": "13",
    "Stone1": "14",
    "Container": "15"
}

LANDMARK_STAGE_NAMES = {
    "ConniferCluster": "Tree",
    "GiantPalm": "Palm tree",
    "House": "House",
    "WaterWell": "Well",
    "Mushroom": "Mushroom",
    "Gorilla": "Gorilla",
    "Windmill": "Windmill",
    "WoodBarrel": "Barrel",
    "LionStatue": "Lion Statue",
    "TreasureChest": "Treasure Chest",
    "Rock": "Rock",
    "Stump": "Tree stump",
    "Pillar": "Pillar",
    "Tombstone": "Tombstone",
    "WoodenChair": "Wooden chair",
    "FireHydrant": "Fire hydrant",
    "OilDrum": "Oil drum",
    "TrafficCone": "Traffic cone",
    "Bench": "Bench",
    "StreetLamp": "Streetlight",
    "PhoneBox": "Phone booth",
    "Anvil": "Anvil",
    "Ladder": "Ladder",
    "Apple": "Apple",
    "Banana": "Banana",
    "RedFlowers": "Purple flowers",
    "YellowFlowers": "Yellow flowers",
    "Cactus": "Cactus",
    "RecycleBin": "Trash bin",
    "Boat": "Boat",
    "Barrel": "Barrel",
    "Barrel2": "Barrel",
    "Beacon": "Lighthouse",
    "BigHouse": "House",
    "Boletus": "Mushroom",
    "Box": "Box",
    "BushTree3": "Bush",
    "BushTree2": "Bush",
    "BushTree": "Bush",
    "Coach": "Coach",
    "Column": "Column",
    "Container": "Container",
    "Dumpster": "Dumpster",
    "GoldCone": "Gold cone",
    "House1": "House",
    "House2": "House",
    "Jet": "Airplane",
    "LowPolyTree": "Tree",
    "LpPine": "Tree",
    "Palm1": "Palm",
    "Palm2": "Palm",
    "Palm3": "Palm",
    "Pickup": "Pickup truck",
    "Pumpkin": "Pumpkin",
    "Soldier": "Soldier",
    "SteelCube": "Cube",
    "Stone1": "Rock",
    "Stone2": "Rock",
    "Stone3": "Rock",
    "Tank": "Tank",
    "Tower2": "Tower",
    "TvTower": "TV Tower",
    "Well": "Well"
}

UNSEEN_LANDMARK_RADII = {
    "FireTruck": 100,
    "Watermelon": 100,
    "Strawberry": 100,
    "Flowerpot": 100,
    "YellowBrick": 100,
    "RedBrick": 100,
    "Globe": 100,
    "Plates": 100,
}

UNSEEN_LANDMARK_STAGE_NAMES = {
    "FireTruck": "21",
    "Watermelon": "22",
    "Strawberry": "23",
    "Flowerpot": "24",
    "YellowBrick": "25",
    "RedBrick": "26",
    "Globe": "27",
    "Plates": "28",
}

if INCLUDE_UNSEEN:
    NUM_LANDMARKS = 74
    LANDMARK_RADII = {**LANDMARK_RADII, **UNSEEN_LANDMARK_RADII}
    LANDMARK_STAGE_NAMES = {**LANDMARK_STAGE_NAMES, **UNSEEN_LANDMARK_STAGE_NAMES}
    ORIG_PORTABLE_LANDMARK_RADII = PORTABLE_LANDMARK_RADII
    PORTABLE_LANDMARK_RADII = {**PORTABLE_LANDMARK_RADII, **UNSEEN_LANDMARK_RADII}
    PORTABLE_LANDMARK_STAGE_NAMES = {**PORTABLE_LANDMARK_STAGE_NAMES, **UNSEEN_LANDMARK_STAGE_NAMES}
else:
    ORIG_PORTABLE_LANDMARK_RADII = PORTABLE_LANDMARK_RADII


def get_landmark_names(add_empty=False):
    if add_empty:
        LANDMARK_RADII[get_null_landmark_name()] = 75
    names = list(sorted(LANDMARK_RADII.keys()))
    return names


def get_null_landmark_name():
    return "0Null"


def get_landmark_name_to_index(single_index=True, add_empty=False):
    """
    :param single_index: If true, return only one index for each landmark name. Otherwise return a list of indices
    :return:
    """
    indices = {}
    landmark_names = get_landmark_names(add_empty)
    for i, name in enumerate(landmark_names):
        if single_index:
            indices[name] = i
        else:
            if name not in indices:
                indices[name] = []
            indices[name].append(i)
    return indices


def get_landmark_index_to_name(add_empty=False):
    name2index = get_landmark_name_to_index(add_empty=add_empty)
    names = name2index.keys()
    indices = name2index.values()
    index_to_name = dict(zip(indices, names))
    for i in range(NUM_LANDMARKS):
        if i not in index_to_name:
            index_to_name[i] = "0Null"
    return index_to_name


def get_landmark_stage_name(name):
    if name in PORTABLE_LANDMARK_STAGE_NAMES:
        return PORTABLE_LANDMARK_STAGE_NAMES[name]
    else:
        return None
