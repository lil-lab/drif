from learning.training.trainer_supervised_bidomain import TrainerBidomain
from learning.training.trainer_supervised_bidomain_bidata import TrainerBidomainBidata
from data_io.models import load_model
from data_io.model_io import save_pytorch_model, load_pytorch_model
from data_io.instructions import get_restricted_env_id_lists

import parameters.parameter_server as P


# Supervised learning parameters
def train_supervised_bidomain():
    P.initialize_experiment()

    setup = P.get_current_parameters()["Setup"]
    supervised_params = P.get_current_parameters()["Supervised"]
    num_epochs = supervised_params["num_epochs"]

    model_sim, _ = load_model(setup["model"], setup["sim_model_file"], domain="sim")
    model_real, _ = load_model(setup["model"], setup["real_model_file"], domain="real")
    model_critic, _ = load_model(setup["critic_model"], setup["critic_model_file"])

    if P.get_current_parameters()["Training"].get("use_oracle_critic", False):
        model_oracle_critic, _ = load_model(setup["critic_model"], setup["critic_model_file"])
        # This changes the name in the summary writer to get a different color plot
        oname = model_oracle_critic.model_name
        model_oracle_critic.set_model_name(oname + "_oracle")
        model_oracle_critic.model_name = oname
    else:
        model_oracle_critic = None

    print("Loading data")
    train_envs, dev_envs, test_envs = get_restricted_env_id_lists()

    real_filename = f"supervised_{setup['model']}_{setup['run_name']}_real"
    sim_filename  = f"supervised_{setup['model']}_{setup['run_name']}_sim"
    critic_filename = f"supervised_{setup['critic_model']}_{setup['run_name']}_critic"

    # TODO: (Maybe) Implement continuing of training

    # Bidata means that we treat Lani++ and LaniOriginal examples differently, only computing domain-adversarial stuff on Lani++
    bidata = P.get_current_parameters()["Training"].get("bidata", False)
    if bidata == "v2":
        trainer = TrainerBidomainBidata(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)
        train_envs_common = [e for e in train_envs if 6000 <= e < 7000]
        train_envs_sim = train_envs
        dev_envs_common = [e for e in dev_envs if 6000 <= e < 7000]
        dev_envs_sim = dev_envs
    elif bidata:
        trainer = TrainerBidomainBidata(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)
        train_envs_common = [e for e in train_envs if 6000 <= e < 7000]
        train_envs_sim = [e for e in train_envs if e < 6000]
        dev_envs_common = [e for e in dev_envs if 6000 <= e < 7000]
        dev_envs_sim = [e for e in dev_envs if e < 6000]
    else:
        trainer = TrainerBidomain(model_real, model_sim, model_critic, model_oracle_critic, epoch=0)

    print("Beginning training...")
    best_test_loss = 1000
    for epoch in range(num_epochs):
        if bidata:
            train_loss = trainer.train_epoch(env_list_common=train_envs_common, env_list_sim=train_envs_sim, eval=False)
            test_loss = trainer.train_epoch(env_list_common=dev_envs_common, env_list_sim=dev_envs_sim, eval=True)
        else:
            train_loss = trainer.train_epoch(env_list=train_envs, eval=False)
            test_loss = trainer.train_epoch(env_list=dev_envs, eval=True)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_pytorch_model(model_real, real_filename)
            save_pytorch_model(model_sim, sim_filename)
            save_pytorch_model(model_critic, critic_filename)
            print(f"Saved models in: \n Real: {real_filename} \n Sim: {sim_filename} \n Critic: {critic_filename}")

        print ("Epoch", epoch, "train_loss:", train_loss, "test_loss:", test_loss)
        save_pytorch_model(model_real, f"tmp/{real_filename}_epoch_{epoch}")
        save_pytorch_model(model_sim, f"tmp/{sim_filename}_epoch_{epoch}")
        save_pytorch_model(model_critic, f"tmp/{critic_filename}_epoch_{epoch}")


if __name__ == "__main__":
    train_supervised_bidomain()
