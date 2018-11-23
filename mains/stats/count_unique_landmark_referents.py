from env_config.definitions.nlp_templates import N_LANDMARKS

if __name__ == "__main__":
    all_referents = []
    for key, reflist in N_LANDMARKS.items():
        all_referents.append(reflist[0])
    print("Num referents: ", len(set(all_referents)))