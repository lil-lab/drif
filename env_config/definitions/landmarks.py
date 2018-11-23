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
    "Mushroom": 130,
    "WoodBarrel": 130,
    "TrafficCone": 130,
    "Apple": 130,
    "Banana": 130,
    "Gorilla": 130,
    "Boat": 130,
    "Coach": 130,
    "GoldCone": 130,
    "LowPolyTree": 130,
    "LpPine": 130,
    "Pumpkin": 130,
    "SteelCube": 130
}

def get_landmark_names(add_empty=False):
    if add_empty:
        LANDMARK_RADII["0Null"] = 75
    names = sorted(list(LANDMARK_RADII.keys()))
    #if real:
    #    names = sorted(LANDMARK_RADII)

    return names


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


def get_landmark_index_to_name():
    name2index = get_landmark_name_to_index()
    names = name2index.keys()
    indices = name2index.values()
    return dict(zip(indices, names))
