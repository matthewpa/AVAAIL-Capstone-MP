import os
from model import model_train, model_load

def main():

    training_data_dir = os.path.join(".", "cs-train")

    ## train the model
    model_train(training_data_dir)

    ## load the model
    model = model_load()

    print("model training complete.")


if __name__ == "__main__":

    main()
