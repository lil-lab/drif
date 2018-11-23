from enum import Enum
import random

T_GOTO__LANDMARK = [
    "go towards {}",
    "go to {}",
    "move to {} and stop"
]


T_GOTO__LANDMARK_SIDE = [
    "go to the {1} side of {0}",
    "go to the {1} of {0}",
    "fly towards {0} and stop on it's {1}",
    "move to {0} and stop on the {1} side of it"
]


T_GOAROUND__LANDMARK_DIR = [
    "move towards {0} and curve {1} around it",
    "go towards {0} and stop behind it, leaving it on the {1} side"
]

T_GOTO__LANDMARK_LANDMARK = [
    "go to {0} and then to {1}"
]

T_GOBETWEEN_LANDMARK_LANDMARK = [
    "go and stop between {0} and {1}"
]


N_LANDMARKS = {
    "ConniferCluster" : ["big tree", "tree", "tall tree"],
    "GiantPalm" : ["palm tree", "big palm", "palm"],
    "House": ["house", "wooden house", "building"],
    "WaterWell": ["well", "water well", "small well"],
    "Mushroom": ["mushroom", "shroom", "red thing"],
    "Gorilla": ["gorilla", "monkey", "ape"],
    "Windmill": ["windmill", "tall building", "wind mill"],
    "WoodBarrel": ["barrel", "wood barrel", "wooden barrel"],
    "LionStatue": ["lion", "stone statue", "lion statue"],
    "TreasureChest": ["chest", "treasure chest", "wooden box"],
    "Rock": ["rock", "stone"],
    "Stump": ["stump", "tree stump"],
    "Pillar": ["pillar", "column", "stone pillar"],
    "Tombstone": ["tombstone", "grave stone", "headstone"],
    "WoodenChair": ["chair", "wooden chair", "big chair"],
    "FireHydrant": ["red hydrant", "hydrant", "fire hydrant"],
    "OilDrum": ["oil drum", "drum", "orange barrel"],
    "TrafficCone": ["cone", "traffic cone", "orange cone"],
    "Bench": ["bench", "seat", "park bench"],
    "StreetLamp": ["lamp", "street lamp", "lantern"],
    "PhoneBox": ["phone booth", "red telephone box", "telephone booth"],
    "Anvil": ["anvil", "metal block"],
    "Ladder": ["ladder", "stairs"],
    "Apple": ["apple", "red apple", "red fruit"],
    "Banana": ["banana", "yellow fruit"],
    "RedFlowers": ["red flowers", "flowers"],
    "YellowFlowers": ["yellow flowers", "flowers"],
    "Cactus": ["cactus", "green tree"],
    "RecycleBin": ["bin", "trash bin", "recycle bin"],
    "Boat": ["boat", "wooden boat", "row boat"],
    "Barrel": ["barrel", "wooden barrel"],
    "Barrel2": ["barrel", "wooden barrel"],
    "Beacon": ["lighthouse", "white building", "beacon"],
    "BigHouse": ["house", "white house", "white house with black roof"],
    "Boletus": ["mushroom", "shroom", "yellow mushroom"],
    "Box": ["box", "wooden box"],
    "BushTree3": ["bush", "bushes", "shrub"],
    "BushTree2": ["bush", "bushes", "shrub"],
    "BushTree": ["bush", "bushes", "shrub"],
    "Coach": ["coach", "bus", "white bus"],
    "Column": ["column", "pillar", "white column"],
    "Container": ["container", "blue container", "blue box"],
    "Dumpster": ["grey dumpster", "grey box", "dumpster"],
    "GoldCone": ["cone", "yellow cone", "golden cone"],
    "House1": ["wooden house", "house"],
    "House2": ["house", "wooden house"],
    "Jet": ["plane", "airplane", "jet"],
    "LowPolyTree": ["christmas tree", "green triangle", "tree"],
    "LpPine": ["christmas tree", "green triangle", "tree"],
    "Palm1": ["palm tree", "palm", "bush"],
    "Palm2": ["palm tree", "palm", "bush"],
    "Palm3": ["palm tree", "palm", "bush"],
    "Pickup": ["truck", "red truck", "red pickup truck"],
    "Pumpkin": ["pumpkin", "big pumpkin"],
    "Soldier": ["statue", "soldier", "green statue"],
    "SteelCube": ["box", "cube"],
    "Stone1": ["rock", "stone"],
    "Stone2": ["rock", "stone"],
    "Stone3": ["rock", "stone"],
    "Tank": ["tank", "combat vehicle"],
    "Tower2": ["tower", "fort"],
    "TvTower": ["tower", "tv tower", "tall tower"],
    "Well": ["well", "big well", "wooden well"]
}

N_SIDES_2 = {
    "left": ["left"],
    "right": ["right"]
}

N_SIDES = {
    "left": ["left"],
    "right": ["right"],
    "front": ["front"],
    "back": ["back"]
}


class TemplateType(Enum):
    GOTO__LANDMARK = 1
    GOTO__LANDMARK_SIDE = 2
    GOAROUND__LANDMARK_DIR = 3
    GOTO__LANDMARK_LANDMARK = 4

    def get_args(self, landmark1, landmark2, side, dir):
        if self == TemplateType.GOTO__LANDMARK:
            return (landmark1,)
        elif self == TemplateType.GOTO__LANDMARK_SIDE:
            return (landmark1, side)
        elif self == TemplateType.GOAROUND__LANDMARK_DIR:
            return (landmark1, dir)
        elif self == TemplateType.GOTO__LANDMARK_LANDMARK:
            return (landmark1, landmark2)

    @staticmethod
    def all():
        return [TemplateType.GOTO__LANDMARK,
                TemplateType.GOTO__LANDMARK_SIDE,
                TemplateType.GOAROUND__LANDMARK_DIR,
                TemplateType.GOTO__LANDMARK_LANDMARK]


def pick_template(template_type, sampling):
    l = None
    if template_type == TemplateType.GOTO__LANDMARK:
        l = T_GOTO__LANDMARK
    elif template_type == TemplateType.GOAROUND__LANDMARK_DIR:
        l = T_GOAROUND__LANDMARK_DIR
    elif template_type == TemplateType.GOTO__LANDMARK_SIDE:
        l = T_GOTO__LANDMARK_SIDE
    elif template_type == TemplateType.GOTO__LANDMARK_LANDMARK:
        l = T_GOTO__LANDMARK_LANDMARK

    opts = len(l)
    if sampling == "random":
        pick = random.randrange(0, opts)
    else:
        pick = 0
    return l[pick]


def pick_option(name, dict, sampling):
    if name is None:
        return None
    l = dict[name]
    if sampling == "random":
        pick = random.randrange(0, len(l))
    elif sampling == "consistent":
        pick = 0
    else:
        print ("Unrecognized sampling method")
        quit(-1)
        return None
    opt = l[pick]
    return opt


def gen_instruction(template_type, landmark1, landmark2, side, dir, sampling="random"):
    template = pick_template(template_type, sampling)
    arg_landmark1 = pick_option(landmark1, N_LANDMARKS, sampling)
    arg_landmark2 = pick_option(landmark2, N_LANDMARKS, sampling)
    arg_side = pick_option(side, N_SIDES, sampling)
    arg_dir = pick_option(dir, N_SIDES, sampling)
    args = template_type.get_args(arg_landmark1, arg_landmark2, arg_side, arg_dir)
    instruction = template.format(*args)
    return instruction


class Template:

    def __init__(self, template_type, landmark1, landmark2=None, side=None, dir=None, sampling="random"):
        self.type = template_type
        self.landmark1 = landmark1
        self.landmark2 = landmark2
        self.side = side
        self.dir = dir
        self.instruction = gen_instruction(template_type, landmark1, landmark2, side, dir, sampling)

    def __str__(self):
        res = "Template: "
        res += str(self.type) + " "
        res += str(self.type.get_args(self.landmark1, self.landmark2, self.side, self.dir)) + " "
        res += self.instruction
        return res


def generate_template(types, landmark_choices=list(N_LANDMARKS.keys()), sampling="random", side_choices=list(N_SIDES.keys())):
    t = random.choice(types)
    landmark1 = random.choice(landmark_choices)
    landmark2 = random.choice(landmark_choices)
    side = random.choice(side_choices)
    dir = random.choice(side_choices)
    return Template(t, landmark1, landmark2, side, dir, sampling)


def generate_templates(count, types=TemplateType.all()):
    out = []
    for i in range(count):
        out.append(generate_template(types))
    return out


def has_ambiguous_noun_phrase(landmark1, landmark2):
    # TODO: This only considers the non-random sampling. Consider random sampling too.
    exp1 = N_LANDMARKS[landmark1][0]
    exp2 = N_LANDMARKS[landmark2][0]
    return exp1 == exp2


def get_side_name2idx():
    keys = list(sorted(list(N_SIDES.keys())))
    name2idx = {}
    for i,key in enumerate(keys):
        name2idx[key] = i
    return name2idx
