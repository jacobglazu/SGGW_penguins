import numpy as np
import pandas as pd
import bentoml
from pydantic import BaseModel

class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    island: str

model_ref = bentoml.sklearn.get("penguins_classifier:latest")
encoder_ref = bentoml.sklearn.get("penguins_encoder:latest")

@bentoml.service(
    name="penguins_service",
    traffic={"timeout": 60},
)

class PenguinsService:
    model = bentoml.sklearn.load_model(model_ref)
    encoder = bentoml.sklearn.load_model(encoder_ref)

    @bentoml.api
    def predict(self, input_data: PenguinFeatures) -> dict:
        data = input_data.model_dump()
        df = pd.DataFrame([data])
        
        cat_features = ["island", "sex"]
        num_features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

        encoded = self.encoder.transform(df[cat_features])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(cat_features))

        final_df = pd.concat([df[num_features].reset_index(drop=True), encoded_df], axis=1)

        prediction = self.model.predict(final_df)
        
        return {"species": str(prediction[0])}