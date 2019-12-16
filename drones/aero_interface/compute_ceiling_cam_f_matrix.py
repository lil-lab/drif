from drones.aero_interface.landmarkConfigurator import LandmarkConfigurator

if __name__ == "__main__":
    cf = LandmarkConfigurator()
    cf.compute_and_save_f_matrix()

