
def dict_zip(list_of_dicts):
    """
    Zip function for dicts!
    Turns a list of dictionaries into a dictionary of lists,
    where each list contains all the elements from the given key in the original list of dicts
    :param list_of_dicts:
    :return:
    """
    outdict = {}
    for dict in list_of_dicts:
        for key,item in dict.items():
            if key not in outdict:
                outdict[key] = []
            outdict[key].append(item)
    return outdict


def dict_map(dict, f, keys=None):
    """
    Applies the given function f to each element in the dict and returns the resulting dict
    :param dict: dict with items as arguments for f
    :param f: function to apply to each element in dict
    :param keys: (optional) list of keys if you'd like to only map f on a set of keys not all of them
    :return: dict of key:ret, where ret = f(dict[key])
    """
    out_dict = dict
    if keys is not None:
        for key in keys:
            item = dict[key]
            try:
                out_dict[key] = f(item)
            except Exception as e:
                print("EXCEPTION! ", e)
    else:
        for key, item in dict.items():
            #print("map: ", key)
            out_dict[key] = f(item)
    return out_dict


def dict_cross_map(dict_a, dict_b, f):
    """
    For each key in dict_a and dict_b, runs f(dict_a[key], dict_a[key]) and keeps the result in out_dict[key].
    Returns out_dict
    :param dict_a:
    :param dict_b:
    :param f:
    :return:
    """
    out_dict = {}
    for key in dict_a:
        if key in dict_b:
            out_dict[key] = f(dict_a[key], dict_b[key])
    return out_dict


def dictlist_append(dict1, dict2):
    """
    For each item2=dict2[key], appends it to dict1[key]
    :param dict1: dict of lists
    :param dict2: dict of items
    :return:
    """
    for key,item in dict2.items():
        if key not in dict1:
            dict1[key] = []
        dict1[key].append(item)
    return dict1


def dict_slice(dict, keys):
    """
    Return a filtered dict that only retains the
    :param dict:
    :param keys:
    :return:
    """
    dict_out = {}
    for key in keys:
        if key in dict:
            dict_out[key] = dict[key]
    return dict_out


def dict_merge(dict1, dict2):
    """
    Recursively merges dict1 and dict2 such that any values in dict2 override values in dict1
    :param dict1:
    :param dict2:
    :return: resulting merged dictionary
    """
    outdict = dict1.copy()
    for k,v in dict2.items():
        # If dict1 has this key and it's also a dictionary, do a recursive merge
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            outdict[k] = dict_merge(dict1[k], v)
        # Otherwise just overwrite the key in dict1
        else:
            outdict[k] = dict2[k]
    return outdict
