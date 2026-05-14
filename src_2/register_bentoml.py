import pickle
import bentoml
import mlflow


def main():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    mod_el = bentoml.sklearn.save_model("penguins_classifier", model)
    enc_der = bentoml.sklearn.save_model("penguins_encoder", encoder)
    
    print(f"Model register in BentoML Store: {mod_el}, {enc_der}")
    
    with mlflow.start_run(run_name="bentoml-registration"):
        mlflow.log_param("classifier_tag", str(mod_el.tag))
        mlflow.log_param("encoder_tag", str(enc_der.tag))

if __name__ == "__main__":
    main()